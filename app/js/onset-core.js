// Phát hiện tiếng "bốp" khi vợt chạm cầu: năng lượng tức thời tăng vọt so với nền.
// Dùng chung cho AudioWorklet và ScriptProcessor (dự phòng).
export class OnsetCore {
  constructor(sampleRate) {
    this.sr = sampleRate;
    this.fast = 0;
    this.slow = 1e-6;
    this.prev = 0;
    this.hp = 0; // bộ lọc thông cao bậc 1 — bỏ tiếng ồn trầm, giữ tiếng đánh cầu
    this.ratio = 7;
    this.floor = 2e-5;
    this.refractory = 0.18 * sampleRate;
    this.sinceHit = Infinity;
    this.warmup = Math.round(0.5 * sampleRate); // nghe nền trước khi bắt đầu phát hiện
    const fc = 2200;
    this.hpA = Math.exp((-2 * Math.PI * fc) / sampleRate);
    this.aFast = 1 - Math.exp(-1 / (0.003 * sampleRate));
    this.aSlow = 1 - Math.exp(-1 / (0.6 * sampleRate));
  }

  setSensitivity(s) {
    // s: 1 (ít nhạy) → 10 (rất nhạy)
    this.ratio = 16 - s * 1.3;
  }

  // Trả về độ mạnh của cú đánh nếu phát hiện trong block này, ngược lại 0.
  process(samples) {
    let hit = 0;
    for (let i = 0; i < samples.length; i++) {
      const x = samples[i];
      this.hp = this.hpA * (this.hp + x - this.prev);
      this.prev = x;
      const e = this.hp * this.hp;
      this.fast += this.aFast * (e - this.fast);
      this.sinceHit++;
      if (this.warmup > 0) {
        this.warmup--;
        this.slow += 4 * this.aSlow * (e - this.slow);
      } else if (
        this.sinceHit > this.refractory &&
        this.fast > this.floor &&
        this.fast > this.ratio * this.slow
      ) {
        hit = Math.max(hit, this.fast / this.slow);
        this.sinceHit = 0;
      }
      // Nền cập nhật chậm, và không bị kéo lên quá nhanh bởi chính tiếng đánh.
      this.slow += this.aSlow * (Math.min(e, this.slow * 4) - this.slow);
      if (this.slow < 1e-9) this.slow = 1e-9;
    }
    return hit;
  }
}
