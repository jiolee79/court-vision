import { COURT } from './court.js';
import { solveHomography, invertH, applyH } from './homography.js';
import { PersonDetector } from './detector.js';
import { Tracker, colorSignature } from './tracker.js';
import { AudioHits, RallyEngine } from './hits.js';
import { drawHud, drawCourtOverlay, formatClock } from './hud.js';
import { HudRecorder, saveBlob } from './recorder.js';

const $ = (id) => document.getElementById(id);
const MAX_SIDE = 1280;     // độ phân giải xử lý/ghi tối đa
const MOTION_W = 256;      // độ rộng ảnh nhỏ dùng đo chuyển động vung vợt

// ---------------- Cài đặt (lưu trên máy) ----------------
const store = {
  get(k, d) { try { const v = localStorage.getItem('cv.' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('cv.' + k, JSON.stringify(v)); } catch { /* chế độ riêng tư */ } },
};

const settings = {
  matchType: store.get('matchType', 'doubles'),
  hitMode: store.get('hitMode', 'audio'),
  quality: store.get('quality', 'fast'),
  sens: store.get('sens', 6),
  names: store.get('names', ['', '', '', '']),
};

function bindSeg(id, key, onChange) {
  const el = $(id);
  const paint = () => el.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === settings[key]));
  el.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    settings[key] = b.dataset.v;
    store.set(key, settings[key]);
    paint();
    onChange?.();
  });
  paint();
  return paint;
}

const paintMatch = () => $('home').classList.toggle('singles', settings.matchType === 'singles');
bindSeg('matchType', 'matchType', paintMatch);
const paintHit1 = bindSeg('hitMode', 'hitMode', () => paintHit2());
const paintHit2 = bindSeg('hitMode2', 'hitMode', () => { paintHit1(); updateHitButton(); });
bindSeg('quality', 'quality');
paintMatch();
for (let i = 0; i < 4; i++) {
  const inp = $('n' + i);
  inp.value = settings.names[i] || '';
  inp.addEventListener('input', () => { settings.names[i] = inp.value.trim(); store.set('names', settings.names); });
}
function syncSens(v) {
  settings.sens = +v;
  store.set('sens', settings.sens);
  $('sens').value = $('sens2').value = v;
  $('sensOut').value = $('sensOut2').value = v;
  S.audioHits?.setSensitivity(settings.sens);
}
$('sens').addEventListener('input', (e) => syncSens(e.target.value));
$('sens2').addEventListener('input', (e) => syncSens(e.target.value));

// ---------------- Trạng thái phiên ----------------
const S = {
  phase: 'idle',      // 'calib' | 'live'
  mode: null,         // 'camera' | 'file'
  video: null,
  stream: null,
  canvas: $('view'),
  ctx: $('view').getContext('2d', { alpha: false }),
  W: 0, H: 0,
  corners: null,      // 4 điểm ảnh: xa-trái, xa-phải, gần-phải, gần-trái
  Hm: null, Hinv: null,
  roi: null,
  detector: null,
  detectorReady: null,
  tracker: null,
  rally: null,
  audioCtx: null,
  audioHits: null,
  recDest: null,
  recorder: new HudRecorder(),
  t0: 0,
  lastFrameT: 0,
  lastDetect: 0,
  detectEvery: 0.08,
  inferMs: 0,
  fps: 0,
  lastVideoTime: -1,
  wakeLock: null,
  showTags: true,
  motion: { canvas: document.createElement('canvas'), prev: null },
};
S.motion.ctx = S.motion.canvas.getContext('2d', { willReadFrequently: true });

function clock() {
  return S.mode === 'file' ? S.video.currentTime : performance.now() / 1000 - S.t0;
}

function playing() {
  return S.mode === 'camera' || (S.video && !S.video.paused && !S.video.ended);
}

// ---------------- UI helpers ----------------
function show(screen) {
  document.querySelectorAll('.screen').forEach((s) => s.classList.toggle('active', s.id === screen));
}
let toastTimer;
function toast(msg, ms = 2200) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), ms);
}
function loading(msg) {
  $('loading').classList.toggle('hidden', !msg);
  if (msg) $('loadingText').textContent = msg;
}

// Canvas hiển thị vừa màn hình, giữ tỉ lệ khung hình.
function fitCanvas() {
  if (!S.W) return;
  const vw = window.innerWidth, vh = window.innerHeight;
  const k = Math.min(vw / S.W, vh / S.H);
  S.canvas.style.width = `${Math.floor(S.W * k)}px`;
  S.canvas.style.height = `${Math.floor(S.H * k)}px`;
}
window.addEventListener('resize', fitCanvas);

// ---------------- Bắt đầu phiên ----------------
$('btnCamera').addEventListener('click', () => startSession('camera'));
$('btnFile').addEventListener('click', () => $('fileInput').click());
$('fileInput').addEventListener('change', (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (f) startSession('file', f);
});

async function startSession(mode, file) {
  // AudioContext phải tạo ngay trong thao tác chạm (yêu cầu của iOS).
  S.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  S.audioCtx.resume();
  S.mode = mode;
  show('stage');
  loading(mode === 'camera' ? 'Đang mở camera…' : 'Đang mở video…');

  const video = document.createElement('video');
  video.playsInline = true;
  video.setAttribute('playsinline', '');
  // Gắn vào trang (ẩn): một số trình duyệt không giải mã video nằm ngoài DOM.
  video.style.cssText = 'position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none';
  document.body.appendChild(video);
  S.video = video;
  S.endShown = false;

  try {
    if (mode === 'camera') {
      video.muted = true;
      S.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } },
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      video.srcObject = S.stream;
      await video.play();
    } else {
      video.src = URL.createObjectURL(file);
      video.preload = 'auto';
      await waitFor(video, ['loadedmetadata'], 20000, 'Không đọc được video này (định dạng không hỗ trợ?)');
      if (video.readyState < 2) {
        // iPhone không tải khung hình nào khi video chưa phát: phát im lặng một chút rồi dừng.
        loading('Đang đọc video…');
        video.muted = true;
        await video.play().catch(() => {});
        await waitFor(video, ['loadeddata', 'canplay'], 10000).catch(() => {});
        video.pause();
        video.muted = false;
      }
      video.currentTime = Math.min(0.5, video.duration / 3 || 0);
      await waitFor(video, ['seeked'], 5000).catch(() => {});
      if (!video.videoWidth) throw new Error('Không đọc được hình ảnh của video này');
      // Hết video → tự mở bảng tổng kết (dùng sự kiện, không phụ thuộc vòng lặp vẽ).
      video.addEventListener('ended', () => {
        if (S.phase !== 'live' || S.endShown) return;
        S.endShown = true;
        S.rally.update(video.currentTime + 60);
        updatePlayBtn();
        renderSummary();
        $('summary').classList.remove('hidden');
      });
    }
  } catch (e) {
    loading('');
    goHome();
    alert(mode === 'camera'
      ? 'Không mở được camera/micro. Hãy cho phép quyền truy cập và mở app qua https.\n\n' + e.message
      : e.message);
    return;
  }

  const k = Math.min(1, MAX_SIDE / Math.max(video.videoWidth, video.videoHeight));
  S.W = S.canvas.width = Math.round(video.videoWidth * k);
  S.H = S.canvas.height = Math.round(video.videoHeight * k);
  fitCanvas();

  // Âm thanh: nguồn → bộ phát hiện tiếng đánh + đường ghi hình (+ loa nếu là video).
  const ctx = S.audioCtx;
  let src = null;
  try {
    src = mode === 'camera' ? ctx.createMediaStreamSource(S.stream) : ctx.createMediaElementSource(video);
    S.recDest = ctx.createMediaStreamDestination();
    src.connect(S.recDest);
    if (mode === 'file') src.connect(ctx.destination);
    S.audioHits = new AudioHits();
    S.audioHits.sensitivity = settings.sens;
    await S.audioHits.start(ctx, src, () => {
      if (S.phase === 'live' && settings.hitMode === 'audio' && playing()) S.rally.addHit(clock(), 'audio');
    });
  } catch (e) {
    console.warn('Âm thanh không khả dụng', e);
    if (settings.hitMode === 'audio') toast('Không đọc được âm thanh — chuyển sang bấm tay', 3000);
    settings.hitMode = 'manual';
    paintHit1(); paintHit2();
  }

  const n = settings.names;
  const names = settings.matchType === 'singles'
    ? [n[0] || 'Người 1', n[2] || 'Người 3']
    : [n[0] || 'Người 1', n[1] || 'Người 2', n[2] || 'Người 3', n[3] || 'Người 4'];
  S.tracker = new Tracker(names, settings.matchType === 'singles');
  S.rally = new RallyEngine(S.tracker);
  S.rally.motionGate = $('motionGate').checked;

  // Tải AI song song trong lúc người dùng hiệu chỉnh sân.
  S.detector = new PersonDetector();
  S.detectorReady = S.detector.load(settings.quality).then(() => true).catch((e) => {
    console.error(e);
    toast('Không tải được mô hình AI (cần Internet lần đầu)', 4000);
    return false;
  });

  loading('');
  enterCalibration();
  requestAnimationFrame(loop);
}

// Chờ một trong các sự kiện của video, có giới hạn thời gian để không bao giờ treo.
function waitFor(el, events, ms, errMsg) {
  return new Promise((resolve, reject) => {
    const done = (fn, arg) => {
      clearTimeout(timer);
      events.forEach((ev) => el.removeEventListener(ev, ok));
      el.removeEventListener('error', fail);
      fn(arg);
    };
    const ok = () => done(resolve);
    const fail = () => done(reject, new Error(errMsg || 'Lỗi đọc video'));
    const timer = setTimeout(() => done(reject, new Error(errMsg || 'Hết thời gian chờ video')), ms);
    events.forEach((ev) => el.addEventListener(ev, ok));
    el.addEventListener('error', fail);
  });
}

function goHome() {
  S.phase = 'idle';
  if (S.recorder.recording) S.recorder.stop();
  S.stream?.getTracks().forEach((t) => t.stop());
  S.stream = null;
  if (S.video) {
    S.video.pause();
    if (S.video.src) URL.revokeObjectURL(S.video.src);
    S.video.remove();
  }
  S.video = null;
  S.audioHits?.stop();
  S.audioCtx?.close();
  S.audioCtx = null;
  S.detector?.close();
  S.wakeLock?.release?.();
  S.wakeLock = null;
  ['calibBar', 'liveBar', 'recBadge', 'sheet', 'summary'].forEach((id) => $(id).classList.add('hidden'));
  show('home');
}

// ---------------- Hiệu chỉnh sân ----------------
function defaultCorners() {
  const { W, H } = S;
  return [[W * 0.36, H * 0.3], [W * 0.64, H * 0.3], [W * 0.88, H * 0.88], [W * 0.12, H * 0.88]];
}

function enterCalibration() {
  S.phase = 'calib';
  if (S.mode === 'file') S.video.pause();
  const saved = store.get('corners', null);
  S.corners = saved ? saved.map(([x, y]) => [x * S.W, y * S.H]) : defaultCorners();
  setCalibZoom(S.corners.some(([x, y]) => x < 0 || y < 0 || x > S.W || y > S.H));
  updateHomography();
  $('calibBar').classList.remove('hidden');
  $('liveBar').classList.add('hidden');
}

function updateHomography() {
  const dst = [[0, 0], [COURT.W, 0], [COURT.W, COURT.L], [0, COURT.L]];
  S.Hm = solveHomography(S.corners, dst);
  S.Hinv = S.Hm && invertH(S.Hm);
}

// Vùng ảnh quanh sân (kèm khoảng ngoài sân và chiều cao người ở phía xa) để đưa vào AI.
function computeRoi() {
  const pts = [];
  for (const [x, y] of [[-1.6, -2.4], [COURT.W + 1.6, -2.4], [COURT.W + 1.6, COURT.L + 2.4], [-1.6, COURT.L + 2.4]]) {
    pts.push(applyH(S.Hinv, x, y));
  }
  let x0 = Math.min(...pts.map((p) => p[0])), x1 = Math.max(...pts.map((p) => p[0]));
  let y0 = Math.min(...pts.map((p) => p[1])), y1 = Math.max(...pts.map((p) => p[1]));
  const farW = Math.hypot(S.corners[1][0] - S.corners[0][0], S.corners[1][1] - S.corners[0][1]);
  y0 -= (farW / COURT.W) * 2.2; // người đứng ở biên xa cao ~2m
  x0 = Math.max(0, x0); y0 = Math.max(0, y0);
  x1 = Math.min(S.W, x1); y1 = Math.min(S.H, y1);
  return [x0, y0, Math.max(16, x1 - x0), Math.max(16, y1 - y0)];
}

// Khi hiệu chỉnh có thể thu nhỏ khung hình để đặt điểm cho góc sân nằm ngoài video.
S.view = { k: 1, ox: 0, oy: 0 };
function setCalibZoom(out) {
  const k = out ? 0.5 : 1;
  S.view = { k, ox: (S.W * (1 - k)) / 2, oy: (S.H * (1 - k)) / 2 };
  $('btnCalibZoom').classList.toggle('on', out);
}
$('btnCalibZoom').addEventListener('click', () => setCalibZoom(S.view.k === 1));

let drag = null;
// Toạ độ chạm → toạ độ ảnh video (đã tính phần thu nhỏ khi hiệu chỉnh).
function toCanvas(e) {
  const r = S.canvas.getBoundingClientRect();
  const { k, ox, oy } = S.view;
  const px = ((e.clientX - r.left) / r.width) * S.W;
  const py = ((e.clientY - r.top) / r.height) * S.H;
  return [(px - ox) / k, (py - oy) / k, S.W / r.width / k];
}
S.canvas.addEventListener('pointerdown', (e) => {
  if (S.phase !== 'calib') return;
  const [x, y, k] = toCanvas(e);
  let best = -1, bd = 70 * k;
  S.corners.forEach(([cx, cy], i) => { const d = Math.hypot(cx - x, cy - y); if (d < bd) { bd = d; best = i; } });
  if (best < 0) return;
  S.canvas.setPointerCapture(e.pointerId);
  drag = { i: best, sx: x, sy: y, ox: S.corners[best][0], oy: S.corners[best][1] };
});
S.canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const [x, y] = toCanvas(e);
  // Kéo tương đối để ngón tay không che điểm đang chỉnh.
  S.corners[drag.i] = [
    Math.min(S.W * 1.5, Math.max(-S.W * 0.5, drag.ox + (x - drag.sx))),
    Math.min(S.H * 1.5, Math.max(-S.H * 0.5, drag.oy + (y - drag.sy))),
  ];
  updateHomography();
});
const endDrag = () => { drag = null; };
S.canvas.addEventListener('pointerup', endDrag);
S.canvas.addEventListener('pointercancel', endDrag);

$('btnCalibReset').addEventListener('click', () => { S.corners = defaultCorners(); updateHomography(); });
$('btnCalibDone').addEventListener('click', async () => {
  if (!S.Hm || !S.Hinv) { toast('4 điểm chưa hợp lệ, hãy kéo lại'); return; }
  store.set('corners', S.corners.map(([x, y]) => [x / S.W, y / S.H]));
  S.tracker.setHomography(S.Hm);
  S.roi = computeRoi();
  setCalibZoom(false);
  $('calibBar').classList.add('hidden');
  loading('Đang tải mô hình AI… (lần đầu cần mạng, khoảng 10–30 giây)');
  const ok = await S.detectorReady;
  loading('');
  if (!S.video) return;
  S.phase = 'live';
  $('liveBar').classList.remove('hidden');
  $('seekRow').classList.toggle('hidden', S.mode !== 'file');
  updateHitButton();
  if (ok) $('perf').textContent = `AI: ${S.detector.delegate}`;
  if (S.mode === 'file') { S.video.play(); updatePlayBtn(); }
  requestWakeLock();
});

// ---------------- Vòng lặp chính ----------------
function loop() {
  if (S.phase === 'idle' || !S.video) return;
  requestAnimationFrame(loop);
  const ctx = S.ctx;
  const nowWall = performance.now() / 1000;
  const dtWall = Math.min(0.1, nowWall - (S.lastFrameT || nowWall));
  S.lastFrameT = nowWall;
  if (dtWall > 0) S.fps += 0.05 * (1 / dtWall - S.fps);

  const s = Math.min(S.W, S.H) / 720;

  if (S.phase === 'calib') {
    const { k, ox, oy } = S.view;
    ctx.fillStyle = '#0b1218';
    ctx.fillRect(0, 0, S.W, S.H);
    ctx.setTransform(k, 0, 0, k, ox, oy);
    ctx.drawImage(S.video, 0, 0, S.W, S.H);
    if (k < 1) {
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 2 / k;
      ctx.strokeRect(0, 0, S.W, S.H);
    }
    if (S.Hinv) drawCourtOverlay(ctx, S.Hinv, s / k);
    drawHandles(ctx, s / k);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return;
  }

  ctx.drawImage(S.video, 0, 0, S.W, S.H);

  const now = clock();
  const newFrame = S.mode === 'camera' || S.video.currentTime !== S.lastVideoTime;
  S.lastVideoTime = S.video.currentTime;

  if (newFrame) {
    updateMotion(now);
    if (S.detector.detector && nowWall - S.lastDetect >= S.detectEvery) {
      S.lastDetect = nowWall;
      const t1 = performance.now();
      const dets = S.detector.detect(S.canvas, S.roi);
      const ms = performance.now() - t1;
      S.inferMs += 0.2 * (ms - S.inferMs);
      // Tự điều chỉnh tần suất nhận diện theo sức mạnh máy: AI chiếm tối đa ~40% thời gian.
      S.detectEvery = Math.min(0.25, Math.max(0.05, (S.inferMs / 1000) * 2.5));
      addSignatures(dets);
      S.tracker.update(dets, now);
    }
  }
  if (playing()) S.tracker.tick(dtWall * (S.mode === 'file' ? S.video.playbackRate : 1));
  S.rally.update(now);

  drawHud(ctx, S.W, S.H, { tracker: S.tracker, rally: S.rally, now, showTags: S.showTags });

  if (S.mode === 'file') {
    updateSeek();
  }
  if (nowWall - (S.lastPerfUi || 0) > 1) {
    S.lastPerfUi = nowWall;
    const v = S.video;
    const progress = S.mode === 'file' && v.duration
      ? `Đã phân tích ${Math.round((v.currentTime / v.duration) * 100)}% · còn ~${fmtMin((v.duration - v.currentTime) / v.playbackRate)} · `
      : '';
    $('perf').textContent = `${progress}AI ${S.detector.delegate || '…'} · ${Math.round(1 / S.detectEvery)} lần/giây`;
  }
}

// Màu áo của từng người phát hiện được (vùng ngực), lấy mẫu 8x12 điểm ảnh — rất rẻ.
const sigCanvas = document.createElement('canvas');
sigCanvas.width = 8;
sigCanvas.height = 12;
const sigCtx = sigCanvas.getContext('2d', { willReadFrequently: true });
function addSignatures(dets) {
  for (const d of dets) {
    sigCtx.drawImage(S.canvas, d.x + d.w * 0.25, d.y + d.h * 0.18, d.w * 0.5, d.h * 0.35, 0, 0, 8, 12);
    d.sig = colorSignature(sigCtx.getImageData(0, 0, 8, 12).data);
  }
}

function drawHandles(ctx, s) {
  const labels = ['1', '2', '3', '4'];
  ctx.strokeStyle = 'rgba(63,208,255,0.9)';
  ctx.lineWidth = 1.5 * s;
  ctx.setLineDash([6 * s, 6 * s]);
  ctx.beginPath();
  S.corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);
  S.corners.forEach(([x, y], i) => {
    const active = drag?.i === i;
    ctx.beginPath();
    ctx.arc(x, y, (active ? 22 : 16) * s, 0, Math.PI * 2);
    ctx.fillStyle = active ? 'rgba(63,208,255,0.35)' : 'rgba(0,0,0,0.45)';
    ctx.fill();
    ctx.lineWidth = 2.5 * s;
    ctx.strokeStyle = '#3fd0ff';
    ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, 2.5 * s, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill();
    ctx.font = `800 ${14 * s}px sans-serif`;
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.fillText(labels[i], x, y - 22 * s);
  });
  if (drag) drawLoupe(ctx, S.corners[drag.i], s);
}

// Kính lúp khi kéo điểm, giúp đặt chính xác vào góc sân.
function drawLoupe(ctx, [x, y], s) {
  const r = 70 * s, zoom = 3;
  const lx = Math.min(S.W - r - 6, Math.max(r + 6, x));
  const ly = y - r - 50 * s > r ? y - r - 50 * s : y + r + 50 * s;
  ctx.save();
  ctx.beginPath(); ctx.arc(lx, ly, r, 0, Math.PI * 2); ctx.clip();
  const sw = (r * 2) / zoom;
  const vk = S.video.videoWidth / S.W;
  ctx.drawImage(S.video, (x - sw / 2) * vk, (y - sw / 2) * vk, sw * vk, sw * vk, lx - r, ly - r, r * 2, r * 2);
  ctx.strokeStyle = '#ff4d5e'; ctx.lineWidth = 1.5 * s;
  ctx.beginPath(); ctx.moveTo(lx - 12 * s, ly); ctx.lineTo(lx + 12 * s, ly); ctx.moveTo(lx, ly - 12 * s); ctx.lineTo(lx, ly + 12 * s); ctx.stroke();
  ctx.restore();
  ctx.beginPath(); ctx.arc(lx, ly, r, 0, Math.PI * 2); ctx.strokeStyle = '#3fd0ff'; ctx.lineWidth = 3 * s; ctx.stroke();
}

// Đo chuyển động phần thân trên của từng người (dùng để biết ai vừa vung vợt).
function updateMotion(now) {
  const M = S.motion;
  const [rx, ry, rw, rh] = S.roi;
  const mw = MOTION_W, mh = Math.max(8, Math.round((rh / rw) * MOTION_W));
  if (M.canvas.width !== mw || M.canvas.height !== mh) { M.canvas.width = mw; M.canvas.height = mh; M.prev = null; }
  M.ctx.drawImage(S.canvas, rx, ry, rw, rh, 0, 0, mw, mh);
  const data = M.ctx.getImageData(0, 0, mw, mh).data;
  const gray = new Uint8Array(mw * mh);
  for (let i = 0, j = 0; j < gray.length; i += 4, j++) gray[j] = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
  const prev = M.prev;
  M.prev = gray;
  if (!prev) return;
  const k = mw / rw;
  for (const p of S.tracker.players) {
    if (!p.visible(now)) continue;
    const b = p.bbox;
    const x0 = Math.max(0, Math.floor((b.x - b.w * 0.35 - rx) * k));
    const x1 = Math.min(mw, Math.ceil((b.x + b.w * 1.35 - rx) * k));
    const y0 = Math.max(0, Math.floor((b.y - b.h * 0.15 - ry) * k));
    const y1 = Math.min(mh, Math.ceil((b.y + b.h * 0.6 - ry) * k));
    let sum = 0, n = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = x0, i = y * mw + x0; x < x1; x++, i++) { sum += Math.abs(gray[i] - prev[i]); n++; }
    }
    const e = n ? sum / n / 255 : 0;
    p.motion.push({ t: now, e });
    while (p.motion.length && p.motion[0].t < now - 1) p.motion.shift();
    p.motionBase += 0.03 * (e - p.motionBase);
  }
}

// ---------------- Điều khiển khi đang phân tích ----------------
function updateHitButton() {
  $('btnHit').textContent = settings.hitMode === 'audio' ? '🎧 Đang nghe · +1' : '🏸 Cú đánh';
}
$('btnHit').addEventListener('pointerdown', (e) => {
  e.preventDefault();
  if (S.phase !== 'live') return;
  S.rally.addHit(clock(), 'manual');
  navigator.vibrate?.(15);
});

function fmtMin(sec) {
  const t = Math.max(0, Math.round(sec));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
}

function updatePlayBtn() { $('btnPlay').textContent = S.video?.paused ? '▶' : '⏸'; }
$('btnPlay').addEventListener('click', () => {
  if (!S.video) return;
  if (S.video.paused) {
    if (S.video.ended) S.endShown = false;
    S.video.play();
  } else {
    S.video.pause();
  }
  updatePlayBtn();
});

// Phân tích nhanh hơn thời gian thực: video phát 1×/2×/3×.
// Tắt giữ cao độ để tiếng đánh cầu không bị làm nhoè khi tua nhanh.
const SPEEDS = [1, 2, 3];
$('btnSpeed').addEventListener('click', () => {
  if (!S.video) return;
  const next = SPEEDS[(SPEEDS.indexOf(S.video.playbackRate) + 1) % SPEEDS.length] || 1;
  S.video.preservesPitch = false;
  S.video.webkitPreservesPitch = false;
  S.video.playbackRate = next;
  $('btnSpeed').textContent = `${next}×`;
  if (next > 1) toast(`Phân tích ${next}× — nhanh hơn nhưng có thể sót vài chi tiết`, 2500);
});
let seeking = false;
$('seek').addEventListener('input', (e) => {
  seeking = true;
  S.video.currentTime = (e.target.value / 1000) * S.video.duration;
});
$('seek').addEventListener('change', () => { seeking = false; });
function updateSeek() {
  const v = S.video;
  if (!seeking && v.duration) $('seek').value = Math.round((v.currentTime / v.duration) * 1000);
  const t = Math.floor(v.currentTime);
  $('timeOut').textContent = `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
  if (v.ended) updatePlayBtn();
}

let recTimer;
$('btnRec').addEventListener('click', async () => {
  if (S.phase !== 'live') return;
  if (!S.recorder.recording) {
    try {
      S.recorder.start(S.canvas, S.recDest?.stream.getAudioTracks()[0]);
    } catch (e) {
      toast('Máy không hỗ trợ ghi hình: ' + e.message, 3500);
      return;
    }
    $('btnRec').classList.add('on');
    $('recBadge').classList.remove('hidden');
    recTimer = setInterval(() => {
      const t = Math.floor((Date.now() - S.recorder.startedAt) / 1000);
      $('recTime').textContent = `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
    }, 500);
  } else {
    clearInterval(recTimer);
    $('btnRec').classList.remove('on');
    $('recBadge').classList.add('hidden');
    const blob = await S.recorder.stop();
    if (blob) {
      toast('Đang lưu video…');
      await saveBlob(blob, `court-vision-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}`);
    }
  }
});

// Cài đặt
$('btnSettings').addEventListener('click', () => {
  $('sheet').classList.remove('hidden');
  syncSens(settings.sens);
  paintHit2();
});
$('sheetClose').addEventListener('click', () => $('sheet').classList.add('hidden'));
$('sheet').addEventListener('click', (e) => { if (e.target.id === 'sheet') $('sheet').classList.add('hidden'); });
$('motionGate').addEventListener('change', (e) => { if (S.rally) S.rally.motionGate = e.target.checked; });
$('showTags').addEventListener('change', (e) => { S.showTags = e.target.checked; });
$('swapNear').addEventListener('click', () => { S.tracker.swapTeammates(0); toast('Đã đổi tên đội gần'); });
$('swapFar').addEventListener('click', () => { S.tracker.swapTeammates(1); toast('Đã đổi tên đội xa'); });
$('swapEnds').addEventListener('click', () => { S.tracker.swapEnds(); S.rally.reset(); toast('Hai đội đã đổi sân'); });
$('recalib').addEventListener('click', () => { $('sheet').classList.add('hidden'); enterCalibration(); });

// Kết thúc & tổng kết
$('btnEnd').addEventListener('click', () => {
  if (S.mode === 'file') { S.video.pause(); updatePlayBtn(); }
  renderSummary();
  $('summary').classList.remove('hidden');
});
$('sumBack').addEventListener('click', () => $('summary').classList.add('hidden'));
$('sumHome').addEventListener('click', goHome);
$('sumShare').addEventListener('click', async () => {
  const txt = summaryText();
  if (navigator.share) {
    try { await navigator.share({ title: 'Court Vision', text: txt }); return; } catch { /* người dùng huỷ */ }
  }
  try { await navigator.clipboard.writeText(txt); toast('Đã sao chép kết quả'); } catch { toast('Không chia sẻ được'); }
});

function rallyStats() {
  const h = S.rally.history;
  const longest = h.reduce((m, r) => Math.max(m, r.count), 0);
  const longestT = h.reduce((m, r) => Math.max(m, r.duration), 0);
  const avg = h.length ? h.reduce((a, r) => a + r.count, 0) / h.length : 0;
  return { n: h.length, longest, longestT, avg };
}

function renderSummary() {
  const r = rallyStats();
  $('sumRallies').innerHTML = `
    <div><b>${r.n}</b><span>Rally</span></div>
    <div><b>${r.longest}</b><span>Rally dài nhất (cú)</span></div>
    <div><b>${r.avg.toFixed(1)}</b><span>Trung bình cú/rally</span></div>`;
  $('sumBody').innerHTML = S.tracker.players.map((p) => `
    <tr><td style="color:${p.color}">${escapeHtml(p.name)}</td>
    <td>${p.dist.toFixed(0)} m</td>
    <td>${(p.maxSpeed * 3.6).toFixed(0)} km/h</td>
    <td>${p.hits}</td>
    <td>${p.maxShot ? p.maxShot + ' km/h' : '–'}</td></tr>`).join('');
}

function summaryText() {
  const r = rallyStats();
  const lines = [`🏸 Court Vision — ${r.n} rally, dài nhất ${r.longest} cú (${formatClock(r.longestT)})`];
  for (const p of S.tracker.players) {
    lines.push(`${p.name}: chạy ${p.dist.toFixed(0)} m, ${p.hits} cú đánh${p.maxShot ? `, cầu nhanh nhất ${p.maxShot} km/h` : ''}`);
  }
  return lines.join('\n');
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// Giữ màn hình sáng khi đang quay.
async function requestWakeLock() {
  try { S.wakeLock = await navigator.wakeLock?.request('screen'); } catch { /* không hỗ trợ */ }
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && S.phase === 'live') requestWakeLock();
});

// ---------------- PWA ----------------
if ('serviceWorker' in navigator && window.isSecureContext) {
  navigator.serviceWorker.register('sw.js').catch((e) => console.warn('SW', e));
}
const isStandalone = matchMedia('(display-mode: standalone)').matches || navigator.standalone;
if (!isStandalone) {
  const ios = /iPhone|iPad|iPod/.test(navigator.userAgent);
  $('installTip').textContent = ios
    ? 'Cài như app: bấm nút Chia sẻ ⬆️ → "Thêm vào MH chính".'
    : 'Cài như app: mở menu trình duyệt ⋮ → "Cài đặt ứng dụng" / "Thêm vào màn hình chính".';
}
