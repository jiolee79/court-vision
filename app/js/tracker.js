// Theo dõi người chơi trên mặt sân (toạ độ mét) và tính quãng đường, tốc độ, thể lực.
import { COURT } from './court.js';
import { applyH } from './homography.js';

export const PLAYER_COLORS = ['#4fc3f7', '#7ee081', '#c792ea', '#ffb74d'];

const MISS_COST = 2.6;
const MAX_RUN_SPEED = 9; // m/s — bước nhảy lớn hơn coi như nhận diện nhầm
const COURT_MARGIN_X = 1.6;
const COURT_MARGIN_Y = 2.6;

export class Player {
  constructor(id, name, side, slot) {
    this.id = id;
    this.name = name;
    this.side = side; // 0 = nửa sân gần camera, 1 = nửa sân xa
    this.slot = slot; // 0 = bên trái, 1 = bên phải (theo góc nhìn camera)
    this.color = PLAYER_COLORS[id % PLAYER_COLORS.length];
    this.reset();
  }

  reset() {
    this.pos = null;      // [x, y] mét, đã làm mượt
    this.bbox = null;     // {x, y, w, h} toạ độ khung hình, đã làm mượt
    this.lastSeen = -1e9;
    this.dist = 0;
    this.speed = 0;
    this.maxSpeed = 0;
    this.stamina = 100;
    this.hits = 0;
    this.maxShot = 0;
    this.trail = [];
    this.motion = [];     // [{t, e}] năng lượng chuyển động phần thân trên
    this.motionBase = 0;
  }

  // Phía sân có đúng với đội không (cho phép lấn qua lưới một chút khi bỏ nhỏ).
  onOwnHalf(y, tol = 0.6) {
    return this.side === 0 ? y > COURT.NET - tol : y < COURT.NET + tol;
  }

  visible(t) {
    return this.bbox && t - this.lastSeen < 1.2;
  }
}

export class Tracker {
  constructor(names, singles) {
    const sides = singles ? [0, 1] : [0, 0, 1, 1];
    const slots = singles ? [0, 0] : [0, 1, 0, 1];
    this.players = sides.map((s, i) => new Player(i, names[i] || `P${i + 1}`, s, slots[i]));
    this.H = null; // ảnh → sân
    this.lastT = null;
  }

  setHomography(H) {
    this.H = H;
  }

  // dets: bbox người trong toạ độ khung hình; t: giây.
  update(dets, t) {
    if (!this.H) return;
    const cand = [];
    for (const d of dets) {
      const [cx, cy] = applyH(this.H, d.x + d.w / 2, d.y + d.h);
      if (cx < -COURT_MARGIN_X || cx > COURT.W + COURT_MARGIN_X) continue;
      if (cy < -COURT_MARGIN_Y || cy > COURT.L + COURT_MARGIN_Y) continue;
      cand.push({ ...d, cx, cy });
    }

    const P = this.players;
    const cost = P.map((p) => cand.map((c) => this.matchCost(p, c, t)));
    let best = { c: Infinity, a: null };
    const assign = new Array(P.length).fill(-1);
    const used = new Array(cand.length).fill(false);
    const rec = (i, acc) => {
      if (acc >= best.c) return;
      if (i === P.length) { best = { c: acc, a: assign.slice() }; return; }
      assign[i] = -1;
      rec(i + 1, acc + MISS_COST);
      for (let j = 0; j < cand.length; j++) {
        if (used[j] || cost[i][j] >= MISS_COST) continue;
        used[j] = true; assign[i] = j;
        rec(i + 1, acc + cost[i][j]);
        used[j] = false; assign[i] = -1;
      }
    };
    rec(0, 0);

    const dt = this.lastT == null ? 0 : Math.min(0.5, Math.max(0, t - this.lastT));
    this.lastT = t;
    P.forEach((p, i) => {
      const j = best.a ? best.a[i] : -1;
      if (j >= 0) this.applyMatch(p, cand[j], t, dt);
      else p.speed *= 0.85;
    });
  }

  matchCost(p, c, t) {
    if (!p.onOwnHalf(c.cy)) return Infinity;
    if (!p.pos || t - p.lastSeen > 4) {
      // Chưa từng thấy: ưu tiên người đứng đúng phía trái/phải của mình.
      const teamSize = this.players.filter((q) => q.side === p.side).length;
      if (teamSize === 1) return 1.2;
      // Ở nửa sân xa, trái/phải theo góc nhìn camera vẫn là x nhỏ/lớn.
      const frac = Math.min(1, Math.max(0, c.cx / COURT.W));
      return 1.0 + (p.slot === 0 ? frac : 1 - frac);
    }
    const dt = Math.min(2, t - p.lastSeen);
    const d = Math.hypot(c.cx - p.pos[0], c.cy - p.pos[1]);
    const gate = 1.8 + 3.2 * dt;
    if (d > gate) return Infinity;
    return (d / gate) * MISS_COST * 0.95;
  }

  applyMatch(p, c, t, dt) {
    const raw = [c.cx, c.cy];
    if (!p.pos || t - p.lastSeen > 1.5) {
      p.pos = raw;
      p.bbox = { x: c.x, y: c.y, w: c.w, h: c.h };
    } else {
      const a = 0.55;
      const nx = p.pos[0] + a * (raw[0] - p.pos[0]);
      const ny = p.pos[1] + a * (raw[1] - p.pos[1]);
      const step = Math.hypot(nx - p.pos[0], ny - p.pos[1]);
      if (dt > 0) {
        const v = step / dt;
        if (v < MAX_RUN_SPEED) {
          if (step > 0.03) p.dist += step;
          p.speed += 0.35 * (v - p.speed);
          p.maxSpeed = Math.max(p.maxSpeed, p.speed);
          // Mô hình thể lực đơn giản: tốn sức khi di chuyển/bứt tốc, hồi khi đứng yên.
          p.stamina -= step * 0.22;
          if (p.speed > 3) p.stamina -= (p.speed - 3) * 0.5 * dt;
        }
      }
      p.pos = [nx, ny];
      const b = 0.5;
      p.bbox.x += b * (c.x - p.bbox.x);
      p.bbox.y += b * (c.y - p.bbox.y);
      p.bbox.w += b * (c.w - p.bbox.w);
      p.bbox.h += b * (c.h - p.bbox.h);
    }
    p.lastSeen = t;
    p.trail.push([p.pos[0], p.pos[1]]);
    if (p.trail.length > 45) p.trail.shift();
  }

  // Gọi mỗi khung hình để hồi thể lực.
  tick(dt) {
    for (const p of this.players) {
      if (p.speed < 0.7) p.stamina += 0.9 * dt;
      p.stamina = Math.min(100, Math.max(5, p.stamina));
    }
  }

  swapTeammates(side) {
    const team = this.players.filter((p) => p.side === side);
    if (team.length !== 2) return;
    const [a, b] = team;
    for (const k of ['pos', 'bbox', 'lastSeen', 'trail', 'motion', 'motionBase', 'speed']) {
      [a[k], b[k]] = [b[k], a[k]];
    }
  }

  // Hai đội đổi sân (sau mỗi game).
  swapEnds() {
    for (const p of this.players) {
      p.side = 1 - p.side;
      p.slot = 1 - p.slot;
      p.pos = null;
      p.bbox = null;
      p.trail = [];
      p.lastSeen = -1e9;
    }
  }
}
