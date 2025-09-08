'use client';

import { useEffect, useRef, useState } from 'react';

export default function useVoiceAgent(wsUrl?: string): [string, (on: boolean) => void] {
  const [status, setStatus] = useState('idle');
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
        // 1) Open WS (binary frames)
        if (wsUrl && !wsRef.current) {
          const ws = new WebSocket(wsUrl);
          ws.binaryType = 'arraybuffer';
          wsRef.current = ws;
        }

        // 2) Mic
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return; }
        streamRef.current = stream;

        // 3) AudioContext + Worklet
        const ctx = new AudioContext();
        ctxRef.current = ctx;
        await ctx.audioWorklet.addModule('/worklets/pcm-16k-processor.js');
        const node = new AudioWorkletNode(ctx, 'pcm16k-processor');
        nodeRef.current = node;

        // 4) Pipe mic into worklet; keep graph alive but muted
        const src = ctx.createMediaStreamSource(stream);
        srcRef.current = src;
        const sink = ctx.createGain(); sink.gain.value = 0;
        src.connect(node).connect(sink).connect(ctx.destination);

        // 5) Receive PCM frames from worklet and forward to WS
        node.port.onmessage = (ev) => {
          if (ev.data?.type === 'pcm') {
            const ab = ev.data.buffer as ArrayBuffer; // 640 bytes per 20ms frame
            wsRef.current?.readyState === 1 && wsRef.current.send(ab);
          }
        };

        setStatus(`capturing → 16 kHz PCM`);
      } catch (e: any) {
        setStatus('failed: ' + (e?.message ?? String(e)));
      }
    }

    function stop() {
      nodeRef.current?.port.postMessage({ type: 'flush' });
      nodeRef.current?.disconnect();
      srcRef.current?.disconnect();
      streamRef.current?.getTracks().forEach(t => t.stop());
      ctxRef.current?.close();
      wsRef.current?.close();

      nodeRef.current = null;
      srcRef.current = null;
      streamRef.current = null;
      ctxRef.current = null;
      wsRef.current = null;

      setStatus('stopped');
    }

    if (on) start(); else stop();
    return () => { cancelled = true; if (on) stop(); };
  }, [on, wsUrl]);

  return [status, setOn];
}
