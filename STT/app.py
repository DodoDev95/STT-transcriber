import asyncio
import contextlib
import json
import logging
import re
from collections import deque
from typing import Deque, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from starlette.websockets import WebSocketState
from starlette.middleware.cors import CORSMiddleware

# ===================== Logging =====================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("stt")

# ===================== Config ======================
SAMPLE_RATE = 16000
WINDOW_SECONDS = 8.0  # sliding decode window for responsiveness
PARTIAL_HOP_MS = 300  # how often partial loop attempts a decode
MIN_WARMUP_SEC = 0.5  # wait before first decode
MODEL_NAME = "small"  # try "small" / "medium" for better quality
COMPUTE_TYPE = "float16"  # "float16" for GPU, "int8" for CPU
LANGUAGE = "en"  # set to "hr" for Croatian, or add config logic

# Silence / energy gate
SILENCE_DBFS_GATE = -45.0  # typical good range: -50..-40 dBFS
MIN_SILENCE_SEC = 1.2  # after this long of silence, drop the window
FINAL_SILENCE_SEC = 0.4  # silence to trigger final (faster endpoint)

# Optional: periodic decode while audio is flowing; guarded by energy gate
FORCE_DECODE_SEC = 1.0

# Trimming params for finalization (two-sided trim)
TRIM_GATE_DBFS = -60.0  # frame RMS must exceed this to count as voiced
TRIM_FRAME_SAMPLES = 320  # 20 ms frames at 16 kHz
TRIM_RIGHT_PAD_MS = 250  # keep a little right pad
TRIM_LEFT_PAD_MS = 50  # keep a tiny left pad
TRIM_MIN_UTT_SEC = 0.20  # allow very short finals, but not absurdly short

# Partials control (throttle + stabilize)
MIN_PARTIAL_DELTA = 4  # require +4 chars growth to send a new partial
MIN_PARTIAL_INTERVAL_MS = 250  # at most ~4 partials/sec
MAX_PARTIAL_SEC = 6.0  # cap utterance audio used for partials to last N sec
VAD_SPEECH_PAD_MS = 100  # smaller pad → faster endpoints

# Finals drift control
MAX_FINAL_SEC = 12.0  # cap final audio length (take tail) to reduce drift
FINAL_LOG_PROB_THRESHOLD = -0.2  # stricter -> filter low-confidence junk

# ================== App & Model ====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Optional[WhisperModel] = None


@app.on_event("startup")
def load_model():
    global model
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE_TYPE)
    log.info("Model loaded: %s (%s)", MODEL_NAME, COMPUTE_TYPE)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "language": LANGUAGE}


# ===================== Helpers =====================
def pcm16_to_float32(b: bytes) -> np.ndarray:
    if not b:
        return np.empty(0, dtype=np.float32)
    a = np.frombuffer(b, dtype=np.int16).astype(np.float32)
    return a / 32768.0


def to_seconds(samples: int) -> float:
    return samples / float(SAMPLE_RATE)


def rms_dbfs(x: np.ndarray) -> float:
    """Short-term level in dBFS for gating."""
    if x.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(x**2)))
    if r <= 1e-8:
        return -120.0
    return 20.0 * np.log10(r)


def should_decode(window: np.ndarray) -> bool:
    """Energy gate before we bother Whisper."""
    level = rms_dbfs(window)
    return level >= SILENCE_DBFS_GATE


def take_tail_seconds(x: np.ndarray, seconds: float) -> np.ndarray:
    if x.size == 0:
        return x
    max_samples = int(seconds * SAMPLE_RATE)
    if x.size <= max_samples:
        return x
    return x[-max_samples:]


def two_sided_trim(x: np.ndarray) -> np.ndarray:
    """
    Trim leading and trailing silence by scanning frames where RMS > TRIM_GATE_DBFS.
    Keep small pads so we don't clip phonemes.
    """
    if x.size == 0:
        return x
    n = x.size
    f = TRIM_FRAME_SAMPLES

    voiced = []
    for start in range(0, n, f):
        chunk = x[start : start + f]
        if rms_dbfs(chunk) > TRIM_GATE_DBFS:
            voiced.append(start)
    if not voiced:
        return np.empty(0, dtype=np.float32)

    first = max(voiced[0] - int((TRIM_LEFT_PAD_MS / 1000.0) * SAMPLE_RATE), 0)
    last_end = min(voiced[-1] + f + int((TRIM_RIGHT_PAD_MS / 1000.0) * SAMPLE_RATE), n)
    return x[first:last_end]


def is_meaningful_text(t: str) -> bool:
    """
    True if t contains at least two alphanumeric chars. Filters out '.', ',', '...' etc.
    """
    if not t:
        return False
    stripped = "".join(ch for ch in t if ch.isalnum())
    return len(stripped) >= 2


def dedupe_sentences(text: str) -> str:
    """
    Collapse consecutive identical sentences (case-insensitive).
    """
    parts = re.split(r"([.!?])", text)
    if not parts:
        return text
    sents = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        end = parts[i + 1] if i + 1 < len(parts) else ""
        if chunk:
            sents.append((chunk, end))
    out = []
    prev = None
    for chunk, end in sents:
        key = chunk.lower()
        if prev is None or key != prev:
            out.append(chunk + end)
            prev = key
    return " ".join(out).strip()


async def whisper_decode(
    arr: np.ndarray,
    *,
    use_vad: bool,
    tag: str,
    beam_size: int = 5,
    temperature: float = 0.0,
    log_prob_threshold: float = -1.0,
) -> str:
    """Call faster-whisper with a chosen VAD setting; return text."""
    log.info(
        ">>> [decode:%s] len=%.2fs, level=%.1f dBFS, vad=%s",
        tag,
        to_seconds(arr.size),
        rms_dbfs(arr),
        use_vad,
    )
    segments, _ = model.transcribe(
        arr,
        language=LANGUAGE,
        task="transcribe",
        vad_filter=use_vad,
        vad_parameters=(
            dict(min_silence_duration_ms=500, speech_pad_ms=VAD_SPEECH_PAD_MS)
            if use_vad
            else None
        ),
        beam_size=beam_size,
        condition_on_previous_text=False,
        temperature=temperature,
        compression_ratio_threshold=2.4,
        log_prob_threshold=log_prob_threshold,
    )
    text = "".join(seg.text for seg in segments).strip()
    log.info(">>> [decode:%s] text=%r", tag, text)
    return text


async def finalize_and_send(
    ws: WebSocket,
    source_audio: np.ndarray,
    total_len_samples: int,
    last_partial: str,
) -> Optional[str]:
    """
    Finalization strategy:
      1) Cap length (tail) to reduce drift.
      2) Primary: VAD ON, beam 5, temp 0.0, stricter log_prob_threshold.
      3) Secondary: two-sided trim + NOVAD (temp 0.0), if VAD text empty/meaningless.
      4) Dedupe repeated sentences.
      5) Fallback to last_partial if still not meaningful.
    """
    # 1) Cap length to reduce drift for very long utterances
    if to_seconds(source_audio.size) > MAX_FINAL_SEC:
        source_audio = take_tail_seconds(source_audio, MAX_FINAL_SEC)
    log.info(
        ">>> [finalize] src=%.2fs (capped) level=%.1f dBFS",
        to_seconds(source_audio.size),
        rms_dbfs(source_audio),
    )

    # 2) Primary: VAD ON (conservative)
    text = await whisper_decode(
        source_audio,
        use_vad=True,
        tag="final.vad",
        beam_size=5,
        temperature=0.0,
        log_prob_threshold=FINAL_LOG_PROB_THRESHOLD,
    )

    # 3) Secondary: two-sided trim + NOVAD (only if needed)
    if not is_meaningful_text(text):
        utt = two_sided_trim(source_audio)
        if utt.size > 0 and to_seconds(utt.size) >= TRIM_MIN_UTT_SEC:
            try:
                text = await whisper_decode(
                    utt,
                    use_vad=False,
                    tag="final.novad-trim",
                    beam_size=5,
                    temperature=0.0,
                    log_prob_threshold=FINAL_LOG_PROB_THRESHOLD,
                )
            except Exception:
                log.exception(">>> [finalize] NOVAD final failed")

    # 4) Post-process dedupe
    if text:
        text = dedupe_sentences(text)

    # 5) Fallback to last partial
    if not is_meaningful_text(text) and last_partial:
        log.info(">>> [finalize] fallback → using last_partial=%r", last_partial)
        text = last_partial

    # Send if meaningful and WS open
    if is_meaningful_text(text) and ws.client_state == WebSocketState.CONNECTED:
        try:
            await ws.send_text(json.dumps({"type": "final", "text": text}))
            log.info(">>> [finalize] SENT final=%r", text)
        except Exception:
            log.exception(">>> [finalize] send failed")
    else:
        log.info(">>> [finalize] nothing meaningful to send (empty/punct or WS closed)")

    return text or None


# ================ WebSocket: /ws ===================
@app.websocket("/ws")
async def ws_stt(ws: WebSocket):
    await ws.accept()
    log.info(">>> WS accepted; sending ready + ping")
    await ws.send_text(json.dumps({"type": "info", "message": "ready"}))
    await ws.send_text(json.dumps({"type": "partial", "text": "ping-from-stt"}))
    log.info(">>> sent ready + ping")

    max_samples = int(WINDOW_SECONDS * SAMPLE_RATE)

    # Sliding window (for partials / responsiveness)
    buf: Deque[np.ndarray] = deque()
    total_samples = 0  # samples currently in window

    # Utterance buffer (for finals)
    utter_buf: Deque[np.ndarray] = deque()
    utter_samples = 0  # samples currently in utterance buffer

    seen_text = ""  # last candidate we saw (for logging de-dupe)
    last_partial = ""  # last good partial (for final fallback)
    last_sent_partial = ""  # last partial we actually SENT
    last_partial_sent_time = 0.0  # monotonic time of last SENT partial

    silence_started_at: Optional[float] = None
    last_voice_at: Optional[float] = (
        None  # monotonic time of last detected voice/energy
    )
    utterance_open = False  # we saw voice; expect a final later

    stop = asyncio.Event()

    def clear_buffer():
        nonlocal total_samples, utter_samples, seen_text, utterance_open
        log.info(
            ">>> [clear_buffer] drop win=%.2fs, utt=%.2fs",
            to_seconds(total_samples),
            to_seconds(utter_samples),
        )
        buf.clear()
        total_samples = 0
        utter_buf.clear()
        utter_samples = 0
        seen_text = ""
        utterance_open = False

    async def safe_send(payload: dict):
        if ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                log.exception("safe_send failed")

    # -------- Background partial emitter ----------
    async def emit_partials():
        nonlocal silence_started_at, seen_text, last_voice_at, utterance_open, last_partial, utter_samples, last_sent_partial, last_partial_sent_time
        try:
            while not stop.is_set():
                await asyncio.sleep(PARTIAL_HOP_MS / 1000.0)
                if ws.client_state != WebSocketState.CONNECTED:
                    break
                if to_seconds(total_samples) < MIN_WARMUP_SEC or not buf:
                    continue

                # Snapshot sliding window (energy gating only)
                window = np.concatenate(list(buf))
                now = asyncio.get_event_loop().time()

                # Energy gate
                if not should_decode(window):
                    if silence_started_at is None:
                        silence_started_at = now
                        log.info(">>> [emit_partials] silence started at %.2f", now)
                    sil = now - silence_started_at
                    log.info(
                        ">>> [emit_partials] silence=%.2fs, utt_open=%s, last_voice_at=%s",
                        sil,
                        utterance_open,
                        f"{last_voice_at:.2f}" if last_voice_at else None,
                    )

                    # End-of-utterance → finalize
                    if (
                        utterance_open
                        and last_voice_at is not None
                        and (now - last_voice_at) > FINAL_SILENCE_SEC
                    ):
                        log.info(
                            ">>> [emit_partials] EOU (silence>%.2fs) → finalize",
                            FINAL_SILENCE_SEC,
                        )
                        final_src = (
                            np.concatenate(list(utter_buf))
                            if utter_samples > 0
                            else window
                        )
                        _ = await finalize_and_send(
                            ws,
                            final_src,
                            utter_samples if utter_samples > 0 else total_samples,
                            last_partial,
                        )
                        # reset for next utterance
                        clear_buffer()
                        silence_started_at = None
                        last_sent_partial = ""
                        last_partial_sent_time = 0.0
                        continue

                    # Long silence cleanup
                    if sil > MIN_SILENCE_SEC:
                        log.info(
                            ">>> [emit_partials] long silence>%.2fs → clear",
                            MIN_SILENCE_SEC,
                        )
                        clear_buffer()
                        last_sent_partial = ""
                        last_partial_sent_time = 0.0
                    continue
                else:
                    # We have energy; reset silence timer and mark voice activity
                    if silence_started_at is not None:
                        log.info(
                            ">>> [emit_partials] energy resumed, resetting silence timer"
                        )
                    silence_started_at = None
                    last_voice_at = now
                    if not utterance_open:
                        # opening a new utterance; reset utter buffer and partial tracking
                        utter_buf.clear()
                        utter_samples = 0
                        last_sent_partial = ""
                        last_partial_sent_time = 0.0
                        log.info(">>> [emit_partials] VOICE → open utterance")
                    utterance_open = True

                # Decode partials from utterance buffer (capped) for stability
                if utter_samples > 0:
                    partial_src = np.concatenate(list(utter_buf))
                    partial_src = take_tail_seconds(partial_src, MAX_PARTIAL_SEC)
                else:
                    partial_src = window  # fallback

                try:
                    text = await whisper_decode(
                        partial_src,
                        use_vad=True,
                        tag="partial.vad",
                        beam_size=1,
                        temperature=0.0,
                        log_prob_threshold=-1.0,
                    )
                    log.info(">>> [emit_partials] partial candidate=%r", text)

                    # Throttle & stabilize
                    has_letters = any(c.isalpha() for c in text)
                    if has_letters:
                        growth = len(text) - len(last_sent_partial)
                        interval_ok = (
                            now - last_partial_sent_time
                        ) * 1000.0 >= MIN_PARTIAL_INTERVAL_MS
                        not_shrinking = len(text) >= len(last_sent_partial)

                        if (
                            growth >= MIN_PARTIAL_DELTA
                            and interval_ok
                            and not_shrinking
                            and text != seen_text
                        ):
                            seen_text = text
                            last_partial = text  # for final fallback
                            last_sent_partial = text  # stabilization reference
                            last_partial_sent_time = now
                            await safe_send({"type": "partial", "text": text})
                            log.info(">>> [emit_partials] SENT partial=%r", text)
                        else:
                            if text != seen_text:
                                seen_text = text
                except Exception:
                    log.exception("emit_partials failed")
        except asyncio.CancelledError:
            log.info(">>> [emit_partials] task cancelled")

    task = asyncio.create_task(emit_partials())

    last_decode_at_samples = 0
    frames_seen = 0

    try:
        # ------------ Ingest loop ------------
        while True:
            msg = await ws.receive()

            if msg["type"] != "websocket.receive":
                log.info(">>> [ingest] non-receive message: %s", msg["type"])
                continue

            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                frames_seen += 1

                f32 = pcm16_to_float32(data)
                if f32.size == 0:
                    continue

                # Append to sliding window
                buf.append(f32)
                total_samples += f32.size

                # Append to utterance buffer only while utterance is open
                if utterance_open:
                    utter_buf.append(f32)
                    utter_samples += f32.size

                if frames_seen % 25 == 0:
                    log.info(
                        ">>> [ingest] recv %d frames; last=%d bytes; total_win=%.2fs, utter=%.2fs",
                        frames_seen,
                        len(data),
                        to_seconds(total_samples),
                        to_seconds(utter_samples),
                    )

                # Trim sliding window to size
                while total_samples > max_samples and buf:
                    drop = buf.popleft()
                    total_samples -= drop.size

                # Optional periodic (guarded) decode for smoother partials (window only)
                if (
                    to_seconds(total_samples - last_decode_at_samples)
                    >= FORCE_DECODE_SEC
                ):
                    window = np.concatenate(list(buf))
                    if should_decode(window):
                        try:
                            text = await whisper_decode(
                                window,
                                use_vad=True,
                                tag="ingest.forced",
                                beam_size=1,
                                temperature=0.0,
                                log_prob_threshold=-1.0,
                            )
                            log.info(">>> [ingest] forced decode: %r", text)
                            # Do not send here; let partial loop decide (throttling)
                            if text and text != seen_text:
                                seen_text = text
                                if len(text) >= 4 and any(c.isalpha() for c in text):
                                    last_partial = text  # keep for final fallback
                        except Exception:
                            log.exception("forced decode failed")
                    last_decode_at_samples = total_samples

            elif "text" in msg and msg["text"] is not None:
                # Handle control messages (e.g., flush; future: config)
                try:
                    obj = json.loads(msg["text"])
                except Exception:
                    obj = {}

                log.info(">>> [ingest] text frame: %s", obj)

                if obj.get("type") == "flush":
                    # Explicit flush only clears buffers (we endpoint via silence)
                    clear_buffer()
                    await safe_send({"type": "info", "message": "flushed"})
                    continue

                # (future) if obj.get("type") == "config": handle language, etc.

    except WebSocketDisconnect:
        log.info(">>> client disconnected")
    finally:
        stop.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # No final in finally: we rely on EOU/flush while WS is open
        clear_buffer()
        log.info(">>> [finally] cleanup complete")
