// public/worklets/pcm-16k-processor.js
class PCM16kProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.step = sampleRate / this.targetRate; // e.g. 48000 / 16000 = 3
    this.t = 0;

    // ring buffer for resampled Float32
    this.buf = new Float32Array(16000 * 2);
    this.read = 0;
    this.write = 0;
    this.len = this.buf.length;
    this.seq = 0;

    this.port.onmessage = (e) => {
      if (e.data?.type === "reset") {
        this.t = 0;
        this.read = this.write = 0;
      } else if (e.data?.type === "flush") {
        this.flush();
      }
    };

    // prove the worklet loaded
    this.port.postMessage({ type: "ready", sr: sampleRate });
  }

  available() {
    return this.write >= this.read
      ? this.write - this.read
      : this.len - (this.read - this.write);
  }

  pushSample(s) {
    this.buf[this.write] = s;
    this.write = (this.write + 1) % this.len;
    if (this.write === this.read) this.read = (this.read + 160) % this.len; // drop ~10 ms on overrun
  }

  emitFrames() {
    const FRAME = 320; // 20 ms @ 16 kHz
    while (this.available() >= FRAME) {
      const i16 = new Int16Array(FRAME);
      for (let i = 0; i < FRAME; i++) {
        const s = this.buf[this.read];
        this.read = (this.read + 1) % this.len;
        const c = Math.max(-1, Math.min(1, s));
        i16[i] = c < 0 ? (c * 0x8000) | 0 : (c * 0x7fff) | 0;
      }
      this.port.postMessage(
        { type: "pcm", seq: this.seq++, buffer: i16.buffer },
        [i16.buffer]
      );
    }
  }

  flush() {
    this.emitFrames();
  }

  process(inputs) {
    // inputs => [ [ Float32Array (ch0), Float32Array (ch1), ... ] ]
    const chs = inputs[0];
    if (!chs || chs.length === 0) return true;
    const data = chs[0]; // Float32Array for channel 0
    if (!data || data.length === 0) return true;

    // linear resample deviceRate → 16k using phase accumulator
    let t = this.t;
    const step = this.step;
    const N = data.length;

    while (t < N) {
      const i = t | 0;
      const frac = t - i;
      const x0 = data[i] || 0;
      const x1 = data[i + 1] || x0;
      this.pushSample(x0 + (x1 - x0) * frac);
      t += step;
    }
    this.t = t - N;
    this.emitFrames();
    return true;
  }
}

registerProcessor("pcm16k-processor", PCM16kProcessor);
