import asyncio
import contextlib
import json
import logging
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
WINDOW_SECONDS = 8.0  # sliding decode window (you can try 4.0)
PARTIAL_HOP_MS = 300  # how often the partial loop attempts a decode
MIN_WARMUP_SEC = 0.5  # wait for this much audio before first decode
MODEL_NAME = "base"  # try "small" / "medium" if you have GPU
COMPUTE_TYPE = "float16"  # "float16" for GPU, "int8" for CPU

# Silence / energy gate (prevents hallucinations while quiet)
SILENCE_DBFS_GATE = -45.0  # typical good range: -50..-40 dBFS
MIN_SILENCE_SEC = 1.2  # after this long of silence, drop the buffer

# Optional: periodic decode while audio is flowing; guarded by energy gate
FORCE_DECODE_SEC = 1.0

# ================== App & Model ====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost",  # safe catch-all for local cases
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
    return {"status": "ok", "model": MODEL_NAME}


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


# ================ WebSocket: /ws ===================
@app.websocket("/ws")
async def ws_stt(ws: WebSocket):
    await ws.accept()
    log.info(">>> sending ready + ping")
    await ws.send_text(json.dumps({"type": "info", "message": "ready"}))
    await ws.send_text(json.dumps({"type": "partial", "text": "ping-from-stt"}))
    log.info(">>> sent ready + ping")

    max_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    buf: Deque[np.ndarray] = deque()
    total_samples = 0

    seen_text = ""  # de-dupe partials
    silence_started_at: Optional[float] = None

    stop = asyncio.Event()

    def clear_buffer():
        nonlocal total_samples, seen_text
        buf.clear()
        total_samples = 0
        seen_text = ""

    async def safe_send(payload: dict):
        if ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                pass

    # -------- Background partial emitter ----------
    async def emit_partials():
        nonlocal silence_started_at, seen_text
        try:
            while not stop.is_set():
                await asyncio.sleep(PARTIAL_HOP_MS / 1000.0)
                if ws.client_state != WebSocketState.CONNECTED:
                    break
                if to_seconds(total_samples) < MIN_WARMUP_SEC or not buf:
                    continue

                # Snapshot sliding window
                window = np.concatenate(list(buf))
                # Energy gate
                if not should_decode(window):
                    if silence_started_at is None:
                        silence_started_at = asyncio.get_event_loop().time()
                    if (
                        asyncio.get_event_loop().time() - silence_started_at
                        > MIN_SILENCE_SEC
                    ):
                        clear_buffer()
                    continue
                else:
                    silence_started_at = None

                try:
                    segments, _ = model.transcribe(
                        window,
                        language="en",
                        task="transcribe",
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=500, speech_pad_ms=200
                        ),
                        beam_size=1,
                        condition_on_previous_text=False,
                        temperature=0.0,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                    )
                    text = "".join(seg.text for seg in segments).strip()
                    if text and text != seen_text:
                        seen_text = text
                        if len(text) >= 4 and any(c.isalpha() for c in text):
                            await safe_send({"type": "partial", "text": text})
                except Exception:
                    log.exception("emit_partials failed")
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(emit_partials())

    last_decode_at_samples = 0
    frames_seen = 0

    try:
        # ------------ Ingest loop ------------
        while True:
            data = await ws.receive_bytes()
            frames_seen += 1
            if frames_seen % 25 == 0:
                log.info(
                    "recv %d frames; last=%d bytes; total=%.2fs",
                    frames_seen,
                    len(data),
                    to_seconds(total_samples),
                )

            f32 = pcm16_to_float32(data)
            if f32.size == 0:
                continue

            buf.append(f32)
            total_samples += f32.size

            # Trim to sliding-window size
            while total_samples > max_samples and buf:
                drop = buf.popleft()
                total_samples -= drop.size

            # -------- Optional periodic decode (guarded) --------
            if to_seconds(total_samples - last_decode_at_samples) >= FORCE_DECODE_SEC:
                window = np.concatenate(list(buf))
                if should_decode(window):
                    try:
                        segments, _ = model.transcribe(
                            window,
                            language="en",
                            task="transcribe",
                            vad_filter=True,
                            vad_parameters=dict(
                                min_silence_duration_ms=500, speech_pad_ms=200
                            ),
                            beam_size=1,
                            condition_on_previous_text=False,
                            temperature=0.0,
                            compression_ratio_threshold=2.4,
                            log_prob_threshold=-1.0,
                        )
                        text = "".join(seg.text for seg in segments).strip()
                        log.info("forced decode: %r", text)
                        if text and text != seen_text:
                            seen_text = text
                            if len(text) >= 4 and any(c.isalpha() for c in text):
                                await safe_send({"type": "partial", "text": text})
                    except Exception:
                        log.exception("forced decode failed")
                last_decode_at_samples = total_samples
            # ----------------------------------------------------

    except WebSocketDisconnect:
        log.info("client disconnected")
    finally:
        stop.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # If we still have audio in buffer, try one last decode
        if buf:
            try:
                window = np.concatenate(list(buf))
                if should_decode(window):
                    segments, _ = model.transcribe(
                        window,
                        language="en",
                        task="transcribe",
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=500, speech_pad_ms=200
                        ),
                        beam_size=5,
                        condition_on_previous_text=False,
                        temperature=0.0,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                    )
                    final_text = "".join(seg.text for seg in segments).strip()
                    print(final_text)
                    if final_text and ws.client_state == WebSocketState.CONNECTED:
                        await safe_send({"type": "final", "text": final_text})
            except Exception:
                log.exception("final decode failed")

        clear_buffer()
