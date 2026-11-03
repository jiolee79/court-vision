// Vẽ lớp HUD kiểu truyền hình lên canvas (nằm trong video khi ghi hình).
import { COURT, courtLines } from './court.js';
import { applyH } from './homography.js';

const FONT = '-apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const PANEL = 'rgba(8, 18, 28, 0.74)';
const EDGE = 'rgba(120, 200, 255, 0.35)';
const ACCENT = '#3fd0ff';
const LABEL = 'rgba(200, 225, 245, 0.75)';

export function formatClock(sec) {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`;
}

function rr(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function panel(ctx, x, y, w, h, s, highlight = 0) {
  rr(ctx, x, y, w, h, 8 * s);
  ctx.fillStyle = PANEL;
  ctx.fill();
  ctx.lineWidth = Math.max(1, 1.2 * s);
  ctx.strokeStyle = highlight > 0 ? `rgba(63, 208, 255, ${0.35 + 0.65 * highlight})` : EDGE;
  ctx.stroke();
}

function text(ctx, str, x, y, size, color, weight = 600, align = 'left') {
  ctx.font = `${weight} ${size}px ${FONT}`;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(str, x, y);
}

// Giá trị căn phải, nhãn nhỏ đứng ngay bên trái giá trị.
function labeledValue(ctx, label, value, right, y, size, s) {
  ctx.font = `800 ${size}px ${FONT}`;
  const vw = ctx.measureText(value).width;
  text(ctx, value, right, y, size, '#fff', 800, 'right');
  text(ctx, label, right - vw - 6 * s, y - 1 * s, 9 * s, LABEL, 700, 'right');
}

function statPanel(ctx, x, y, w, h, s, label, value, unit, hl) {
  panel(ctx, x, y, w, h, s, hl);
  text(ctx, label, x + 12 * s, y + 20 * s, 12 * s, LABEL, 700);
  text(ctx, value, x + 12 * s, y + h - 14 * s, 34 * s, '#ffffff', 800);
  if (unit) {
    ctx.font = `800 ${34 * s}px ${FONT}`;
    const vw = ctx.measureText(value).width;
    text(ctx, unit, x + 16 * s + vw, y + h - 15 * s, 12 * s, LABEL, 700);
  }
}

export function drawHud(ctx, W, H, st) {
  const { tracker, rally, now, showTags } = st;
  const s = Math.min(W, H) / 720;
  const pad = 14 * s;
  const gap = 8 * s;
  const hudW = Math.min(W - 2 * pad, 700 * s);

  if (showTags) drawPlayerTags(ctx, tracker, rally, now, s);

  // Hàng trên: thời gian rally, số cú đánh, tốc độ cầu
  const topH = 74 * s;
  const w1 = hudW * 0.4, w2 = hudW * 0.27, w3 = hudW - w1 - w2 - 2 * gap;
  const flash = Math.max(0, 1 - (now - rally.flash) / 0.4);
  statPanel(ctx, pad, pad, w1, topH, s, 'RALLY TIME', formatClock(rally.rallyTime(now)), '', 0);
  statPanel(ctx, pad + w1 + gap, pad, w2, topH, s, 'RALLY COUNT', String(rally.count), 'SHOTS', flash);
  statPanel(ctx, pad + w1 + w2 + 2 * gap, pad, w3, topH, s, 'SHOT SPEED',
    rally.lastShotSpeed ? String(rally.lastShotSpeed) : '--', 'KM/H', 0);

  // Bảng người chơi
  const y2 = pad + topH + gap;
  const tw = hudW * 0.5;
  const rowH = 44 * s;
  const players = tracker.players;
  const th = rowH * players.length + 8 * s;
  panel(ctx, pad, y2, tw, th, s);
  players.forEach((p, i) => {
    const ry = y2 + 4 * s + i * rowH;
    const x0 = pad + 10 * s;
    ctx.fillStyle = p.color;
    ctx.fillRect(pad + 3 * s, ry + 6 * s, 3 * s, rowH - 12 * s);
    text(ctx, p.name.toUpperCase(), x0, ry + 20 * s, 17 * s, p.color, 800);
    labeledValue(ctx, 'STAMINA', String(Math.round(p.stamina)), pad + tw - 10 * s, ry + 20 * s, 17 * s, s);
    // Thanh thể lực dạng vạch
    const bx = x0, by = ry + 27 * s, bw = tw * 0.62, bh = 6 * s;
    const n = 20;
    const filled = Math.round((p.stamina / 100) * n);
    for (let k = 0; k < n; k++) {
      ctx.fillStyle = k < filled ? p.color : 'rgba(255,255,255,0.12)';
      ctx.fillRect(bx + (k * bw) / n, by, bw / n - 1.5 * s, bh);
    }
    labeledValue(ctx, 'MOVE', `${p.dist.toFixed(1)} m`, pad + tw - 10 * s, ry + 36 * s, 12 * s, s);
  });

  // Bản đồ sân nhìn từ trên xuống
  const mx = pad + tw + gap;
  const mw = hudW - tw - gap;
  drawCourtMap(ctx, mx, y2, mw, th, s, tracker, now);
}

function drawCourtMap(ctx, x, y, w, h, s, tracker, now) {
  panel(ctx, x, y, w, h, s);
  text(ctx, 'COURT MAP', x + 10 * s, y + 18 * s, 11 * s, LABEL, 700);
  text(ctx, 'PLAYER POSITIONS', x + w - 10 * s, y + 18 * s, 9 * s, LABEL, 600, 'right');
  const areaX = x + 10 * s, areaY = y + 26 * s, areaW = w - 20 * s, areaH = h - 46 * s;
  // Sân nằm ngang: chiều dài sân theo trục x của bản đồ, đội gần camera bên trái.
  const scale = Math.min(areaW / COURT.L, areaH / COURT.W);
  const cw = COURT.L * scale, ch = COURT.W * scale;
  const ox = areaX + (areaW - cw) / 2, oy = areaY + (areaH - ch) / 2;
  const map = (cx, cy) => [ox + ((COURT.L - cy) / COURT.L) * cw, oy + (cx / COURT.W) * ch];
  ctx.fillStyle = 'rgba(30, 110, 90, 0.35)';
  ctx.fillRect(ox, oy, cw, ch);
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = Math.max(1, s);
  ctx.beginPath();
  for (const [x1, y1, x2, y2] of courtLines()) {
    const a = map(x1, y1), b = map(x2, y2);
    ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
  }
  ctx.stroke();
  const n1 = map(0, COURT.NET), n2 = map(COURT.W, COURT.NET);
  ctx.strokeStyle = ACCENT;
  ctx.lineWidth = 2 * s;
  ctx.beginPath(); ctx.moveTo(n1[0], n1[1] - 3 * s); ctx.lineTo(n2[0], n2[1] + 3 * s); ctx.stroke();

  for (const p of tracker.players) {
    if (!p.pos) continue;
    ctx.strokeStyle = p.color;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1.5 * s;
    ctx.beginPath();
    p.trail.forEach(([tx, ty], i) => {
      const [a, b] = map(tx, ty);
      i ? ctx.lineTo(a, b) : ctx.moveTo(a, b);
    });
    ctx.stroke();
    ctx.globalAlpha = now - p.lastSeen < 1.2 ? 1 : 0.4;
    const [px, py] = map(p.pos[0], p.pos[1]);
    ctx.fillStyle = p.color;
    ctx.beginPath(); ctx.arc(px, py, 5 * s, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2 * s; ctx.stroke();
    ctx.globalAlpha = 1;
  }
  text(ctx, 'OUR TEAM', x + 10 * s, y + h - 8 * s, 9 * s, LABEL, 700);
  text(ctx, 'OPPONENTS', x + w - 10 * s, y + h - 8 * s, 9 * s, LABEL, 700, 'right');
}

function drawPlayerTags(ctx, tracker, rally, now, s) {
  for (const p of tracker.players) {
    if (!p.visible(now)) continue;
    const b = p.bbox;
    const cx = b.x + b.w / 2;
    // Vòng chân
    ctx.strokeStyle = p.color;
    ctx.lineWidth = 2.5 * s;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    ctx.ellipse(cx, b.y + b.h, Math.max(14 * s, b.w * 0.45), Math.max(5 * s, b.w * 0.14), 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
    // Nhãn tên
    const fs = 15 * s;
    ctx.font = `800 ${fs}px ${FONT}`;
    const label = p.name.toUpperCase();
    const tw = ctx.measureText(label).width + 16 * s;
    const th = fs + 10 * s;
    const tx = cx - tw / 2, ty = b.y - th - 10 * s;
    rr(ctx, tx, ty, tw, th, 5 * s);
    ctx.fillStyle = PANEL; ctx.fill();
    ctx.strokeStyle = p.color; ctx.lineWidth = 1.5 * s; ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - 5 * s, ty + th); ctx.lineTo(cx + 5 * s, ty + th); ctx.lineTo(cx, ty + th + 6 * s);
    ctx.closePath(); ctx.fillStyle = p.color; ctx.fill();
    text(ctx, label, cx, ty + th - 7 * s, fs, p.color, 800, 'center');
  }
  const tag = rally.shotTag;
  if (tag && now - tag.t < 1.8 && tag.player.visible(now)) {
    const b = tag.player.bbox;
    const fs = 14 * s;
    const str = `${tag.speed} km/h`;
    ctx.font = `800 ${fs}px ${FONT}`;
    const tw = ctx.measureText(str).width + 50 * s;
    const th = fs + 10 * s;
    const tx = b.x + b.w + 6 * s, ty = b.y + b.h * 0.2;
    rr(ctx, tx, ty, tw, th, 5 * s);
    ctx.fillStyle = PANEL; ctx.fill();
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 1.5 * s; ctx.stroke();
    text(ctx, 'SHOT', tx + 8 * s, ty + th - 8 * s, 9 * s, LABEL, 700);
    text(ctx, str, tx + 42 * s, ty + th - 7 * s, fs, '#fff', 800);
  }
}

// Vẽ các đường sân đã hiệu chỉnh lên khung hình (để kiểm tra độ khớp).
export function drawCourtOverlay(ctx, Hinv, s, alpha = 0.8) {
  ctx.strokeStyle = `rgba(63, 208, 255, ${alpha})`;
  ctx.lineWidth = 2 * s;
  ctx.beginPath();
  for (const [x1, y1, x2, y2] of courtLines()) {
    const a = applyH(Hinv, x1, y1), b = applyH(Hinv, x2, y2);
    ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
  }
  ctx.stroke();
  const n1 = applyH(Hinv, 0, COURT.NET), n2 = applyH(Hinv, COURT.W, COURT.NET);
  ctx.strokeStyle = `rgba(255, 210, 63, ${alpha})`;
  ctx.lineWidth = 3 * s;
  ctx.beginPath(); ctx.moveTo(n1[0], n1[1]); ctx.lineTo(n2[0], n2[1]); ctx.stroke();
}
