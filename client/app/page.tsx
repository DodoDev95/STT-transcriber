'use client';
import useVoiceAgent from './useVoiceAgent';

export default function Page() {
  const ws = process.env['NEXT_PUBLIC_WS_URL'] || 'ws://localhost:8080/ws';
  const [status, setOn] = useVoiceAgent(ws);

  return (
    <main style={{ padding: 24, fontFamily: 'ui-sans-serif, system-ui' }}>
      <h1 style={{ fontSize: 22, fontWeight: 700 }}>Voice Agent (AudioWorklet)</h1>
      <p style={{ opacity: .8 }}>WS: <code>{ws}</code></p>
      <p>{status}</p>
      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={() => setOn(true)}>Start</button>
        <button onClick={() => setOn(false)}>Stop</button>
      </div>
    </main>
  );
}
