'use client';
import useVoiceAgent from './useVoiceAgent';

export default function Page() {
  const ws = process.env['NEXT_PUBLIC_WS_URL'] || 'ws://localhost:8080/ws';
  const [status, partial, transcript, setOn] = useVoiceAgent(ws);

  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-900 p-6 text-gray-100">
      <div className="w-full max-w-xl rounded-2xl bg-gray-800 shadow-lg p-6">
        {/* Status */}
        <p className="text-sm text-gray-400 mb-4">{status}</p>

        {/* Controls */}
        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setOn(true)}
            className="flex-1 rounded-lg bg-green-600 px-4 py-2 text-white font-semibold hover:bg-green-700 active:scale-95 transition"
          >
            Start
          </button>
          <button
            onClick={() => setOn(false)}
            className="flex-1 rounded-lg bg-red-600 px-4 py-2 text-white font-semibold hover:bg-red-700 active:scale-95 transition"
          >
            Stop
          </button>
        </div>

        {/* Live partial */}
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-gray-200 mb-1">Live</h2>
          <div className="rounded-md bg-yellow-900/40 border border-yellow-700 p-2 min-h-[2rem] text-yellow-100">
            {partial || <span className="text-gray-500">…waiting…</span>}
          </div>
        </div>

        {/* Transcript */}
        <div>
          <h2 className="text-lg font-semibold text-gray-200 mb-1">Transcript</h2>
          <div className="rounded-md bg-gray-700 border border-gray-600 p-3 h-40 overflow-y-auto whitespace-pre-wrap text-gray-100">
            {transcript || <span className="text-gray-500">No transcript yet</span>}
          </div>
        </div>
      </div>
    </main>
  );
}
