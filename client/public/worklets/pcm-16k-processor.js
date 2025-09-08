// AudioWorklet that downsamples to 16 kHz, frames 20 ms, and posts Int16 frames
class PCM16kProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.step = sampleRate / this.targetRate; // input samples per 16k output sample
    this.t = 0; // phase accumulator in input-sample units

    // simple circular buffer for resampled float32
    this.buf = new Float32Array(16000 * 2); // 2 seconds of 16k audio
    this.read = 0;
    this.write = 0;
    this.len = this.buf.length;
    this.seq = 0;

    this.port.onmessage = (e) => {
      if (e.data?.type === "reset") {
        this.t = 0;
        this.read = this.write = 0;
      }
      if (e.data?.type === "flush") this.flush();
    };
  }

  // how many samples currently buffered
  available() {
    return this.write >= this.read
      ? this.write - this.read
      : this.len - (this.read - this.write);
  }

  // push one sample into ring buffer
  pushSample(s) {
    this.buf[this.write] = s;
    this.write = (this.write + 1) % this.len;
    // overrun protection: move read forward if we ever overlap
    if (this.write === this.read) this.read = (this.read + 160) % this.len; // drop ~10 ms
  }

  emitFrames() {
    const FRAME = 320; // 20 ms @ 16k
    while (this.available() >= FRAME) {
      const i16 = new Int16Array(FRAME);
      for (let i = 0; i < FRAME; i++) {
        const s = this.buf[this.read];
        this.read = (this.read + 1) % this.len;
        const c = Math.max(-1, Math.min(1, s));
        i16[i] = c < 0 ? c * 0x8000 : c * 0x7fff; // −32768..+32767
      }
      // transfer the underlying buffer to main thread (zero copy)
      this.port.postMessage(
        { type: "pcm", seq: this.seq++, buffer: i16.buffer },
        [i16.buffer]
      );
    }
  }

  flush() {
    // optionally pad leftover to a full frame (not strictly required for STT)
    this.emitFrames();
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch0 = input[0]; // Float32Array of 128 frames at the device sampleRate
    if (!ch0) return true;

    // linear-resample from device rate -> 16 kHz using phase accumulator
    let t = this.t; // current position in input-sample units
    const step = this.step; // input samples / 16k output sample
    const N = ch0.length;

    while (t < N) {
      const i = Math.floor(t);
      const frac = t - i;
      const x0 = ch0[i] || 0;
      const x1 = ch0[i + 1] || x0;
      const y = x0 + (x1 - x0) * frac; // interpolated sample at time t
      this.pushSample(y);
      t += step;
    }

    this.t = t - N; // carry remainder into next render quantum
    this.emitFrames();
    return true; // keep processor alive
  }
}

registerProcessor("pcm16k-processor", PCM16kProcessor);
