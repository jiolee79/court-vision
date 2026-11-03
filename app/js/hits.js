// Phát hiện cú đánh (âm thanh / bấm tay), xác định người đánh, đếm rally và ước tính tốc độ cầu.
import { OnsetCore } from './onset-core.js';

// Cầu lông chịu lực cản rất lớn: vận tốc rơi tới hạn ~6.7 m/s → k = g / vt².
const SHUTTLE_K = 9.81 / (6.7 * 6.7);
const PROCESS_DELAY = 0.12; // chờ thêm chút để thấy chuyển động vung vợt sau tiếng đánh

export class AudioHits {
  constructor() {
    this.node = null;
    this.sink = null;
    this.sensitivity = 6;
  }

  async start(ctx, source, onHit) {
    this.stop();
    this.sink = ctx.createGain();
    this.sink.gain.value = 0;
    this.sink.connect(ctx.destination);
    if (ctx.audioWorklet && window.isSecureContext) {
      await ctx.audioWorklet.addModule(new URL('./onset-worklet.js', import.meta.url));
      this.node = new AudioWorkletNode(ctx, 'onset-processor');
      this.node.port.onmessage = (e) => onHit(e.data.strength);
      this.node.port.postMessage({ sensitivity: this.sensitivity });
    } else {
      // Dự phòng khi không có AudioWorklet (ví dụ mở qua http trong mạng LAN).
      const core = new OnsetCore(ctx.sampleRate);
      core.setSensitivity(this.sensitivity);
      this.core = core;
      this.node = ctx.createScriptProcessor(1024, 1, 1);
      this.node.onaudioprocess = (e) => {
        const s = core.process(e.inputBuffer.getChannelData(0));
        if (s) onHit(s);
      };
    }
    source.connect(this.node);
    this.node.connect(this.sink);
    this.source = source;
  }

  setSensitivity(s) {
    this.sensitivity = s;
    this.node?.port?.postMessage({ sensitivity: s });
    this.core?.setSensitivity(s);
  }

  stop() {
    try { this.source?.disconnect(this.node); } catch { /* đã ngắt */ }
    this.node?.disconnect();
    this.sink?.disconnect();
    this.node = this.sink = this.source = this.core = null;
  }
}

export class RallyEngine {
  constructor(tracker) {
    this.tracker = tracker;
    this.endGap = 2.6;      // giây không có cú đánh → kết thúc rally
    this.motionGate = true; // chỉ nhận tiếng đánh khi có người trên sân vung vợt
    this.pending = [];
    this.history = [];
    this.reset();
  }

  reset() {
    this.active = false;
    this.start = 0;
    this.lastHit = -1e9;
    this.count = 0;
    this.lastDuration = 0;
    this.lastShotSpeed = 0;
    this.shotTag = null;
    this.prev = null;
    this.flash = -1e9;
    this.pending = [];
  }

  addHit(t, source) {
    this.pending.push({ t, source });
  }

  update(now) {
    if (this.active && now < this.lastHit - 0.5) this.reset(); // tua video ngược
    while (this.pending.length && now - this.pending[0].t >= PROCESS_DELAY) {
      const h = this.pending.shift();
      const { player, score } = this.pickHitter(h.t);
      const anyVisible = this.tracker.players.some((p) => p.visible(h.t));
      if (h.source === 'audio' && this.motionGate && anyVisible && score < 1.6) continue;
      this.registerHit(h.t, player);
    }
    if (this.active && now - this.lastHit > this.endGap) {
      this.active = false;
      this.lastDuration = this.lastHit - this.start + 0.5;
      this.history.push({ count: this.count, duration: this.lastDuration, end: this.lastHit });
    }
  }

  pickHitter(t) {
    let best = null;
    let bestScore = 0;
    const prevSide = this.active && this.prev?.player ? this.prev.player.side : null;
    for (const p of this.tracker.players) {
      if (!p.visible(t)) continue;
      let peak = 0;
      for (const m of p.motion) if (m.t >= t - 0.25 && m.t <= t + 0.1) peak = Math.max(peak, m.e);
      let score = peak / (p.motionBase + 0.004);
      // Trong một rally, hai bên luân phiên đánh.
      if (prevSide != null && p.side === prevSide) score *= 0.35;
      if (score > bestScore) { bestScore = score; best = p; }
    }
    return { player: best, score: bestScore };
  }

  registerHit(t, player) {
    if (!this.active || t - this.lastHit > this.endGap) {
      this.active = true;
      this.start = t;
      this.count = 0;
      this.prev = null;
    }
    this.count++;
    this.lastHit = t;
    this.flash = t;
    if (player) {
      player.hits++;
      player.stamina -= 0.25;
    }
    const prev = this.prev;
    if (prev && prev.player && player && prev.pos && player.pos && prev.player.side !== player.side) {
      const x = Math.hypot(player.pos[0] - prev.pos[0], player.pos[1] - prev.pos[1]);
      const dt = t - prev.t;
      if (dt > 0.2 && dt < 3 && x > 1) {
        // Vận tốc ban đầu từ quãng đường và thời gian bay, có tính lực cản: x(t) = ln(1 + k·v0·t) / k
        const v0 = (Math.exp(SHUTTLE_K * x) - 1) / (SHUTTLE_K * dt);
        const kmh = Math.min(420, Math.max(15, v0 * 3.6));
        this.lastShotSpeed = Math.round(kmh);
        prev.player.maxShot = Math.max(prev.player.maxShot, this.lastShotSpeed);
        this.shotTag = { player: prev.player, speed: this.lastShotSpeed, t };
      }
    }
    this.prev = { player, pos: player?.pos ? [...player.pos] : null, t };
  }

  rallyTime(now) {
    return this.active ? Math.max(0, now - this.start) : this.lastDuration;
  }
}
