"use client";

import { useEffect, useRef, useState } from "react";

export default function useVoiceAgent(
  wsUrl?: string
): [status: string, partial: string, transcript: string, setOn: (on: boolean) => void] {
  const [status, setStatus] = useState("idle");
  const [partial, setPartial] = useState("");
  const [transcript, setTranscript] = useState("");
  const [on, setOn] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);
  const srcRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function start() {
      try {
        // 1) Open WS
        if (wsUrl && !wsRef.current) {
          const ws = new WebSocket(wsUrl);
          ws.binaryType = "arraybuffer";

          ws.onopen = () => {
            setStatus("ws connected, capturing → 16 kHz PCM");
          };

          ws.onmessage = (evt) => {
            try {
              const msg = JSON.parse(evt.data);

              if (msg.type === "meter") {
                setStatus(`📈 ${msg.rms_dbfs.toFixed(1)} dBFS (peak ${msg.peak_dbfs.toFixed(1)})  ${msg.seconds.toFixed(2)}s`);
              } else if (msg.type === "info") {
                setStatus(`info: ${msg.message}`);
              } else if (msg.type === "partial") {
                setPartial(msg.text ?? "");
              } else if (msg.type === "final") {
                console.log("transcription", msg);
                const t = (msg.text ?? "").trim();
                if (t) setTranscript((prev) => (prev ? prev + " " + t : t));
                setPartial("");
              }
            } catch {
              // Non-JSON => ignore
            }
          };

          ws.onclose = () => setStatus("ws closed");
          ws.onerror = () => setStatus("ws error");
          wsRef.current = ws;
        }

        // 2) Mic
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false },
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;

        // 3) AudioContext + Worklet
        const ctx = new AudioContext();
        ctxRef.current = ctx;
        await ctx.audioWorklet.addModule("/worklets/pcm-16k-processor.js");
        const node = new AudioWorkletNode(ctx, "pcm16k-processor");
        nodeRef.current = node;

        // 4) Wire up (muted sink keeps graph alive)
        const src = ctx.createMediaStreamSource(stream);
        srcRef.current = src;
        const sink = ctx.createGain();
        sink.gain.value = 0;
        src.connect(node).connect(sink).connect(ctx.destination);

        // 5) Forward 16 kHz PCM16 frames to WS
        node.port.onmessage = (ev) => {
          if (ev.data?.type === "pcm") {
            const ab = ev.data.buffer as ArrayBuffer;
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(ab);
            }
          }
        };

        setStatus(`capturing → 16 kHz PCM`);
      } catch (e: any) {
        setStatus("failed: " + (e?.message ?? String(e)));
      }
    }

    async function stop() {
      // tell the worklet to flush its local ringbuffer (optional but fine)
      nodeRef.current?.port.postMessage({ type: "flush" });

      // *** IMPORTANT: ask the server to finalize before we close ***
      const ws = wsRef.current;
      let gotFinal = false;

      if (ws && ws.readyState === WebSocket.OPEN) {
        // temporary handler to detect final / flushed
        const onMessage = (evt: MessageEvent) => {
          try {
            const msg = JSON.parse(evt.data);
            if (msg.type === "final" || (msg.type === "info" && msg.message === "flushed")) {
              gotFinal = true;
            }
          } catch {}
        };
        ws.addEventListener("message", onMessage);

        // send flush command to server
        ws.send(JSON.stringify({ type: "flush" }));

        // wait up to 600ms for final (tune this)
        await new Promise((resolve) => setTimeout(resolve, 600));

        ws.removeEventListener("message", onMessage);
      }

      // tear down audio regardless
      nodeRef.current?.disconnect();
      srcRef.current?.disconnect();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      await ctxRef.current?.close().catch(() => {});
      // close ws after giving it a chance to send final
      wsRef.current?.close();

      nodeRef.current = null;
      srcRef.current = null;
      streamRef.current = null;
      ctxRef.current = null;
      wsRef.current = null;

      setStatus("stopped");
      setPartial("");
    }

    if (on) start();
    else void stop();

    return () => {
      cancelled = true;
      if (on) void stop();
    };
  }, [on, wsUrl]);

  return [status, partial, transcript, setOn];
}
