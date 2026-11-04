"""Nhận diện người chơi bằng YOLO11-pose (gọi thẳng OpenVINO, chạy song song), rồi gán đúng tên.

Pose cho vị trí cổ chân (điểm chạm sân chính xác hơn đáy khung) và cổ tay (biết ai đang vung vợt).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

import court as C
from paths import MODELS_DIR

# Chỉ số khung xương COCO
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 9, 10, 11, 12, 15, 16
MARGIN_X, MARGIN_Y = 1.0, 2.6  # ngoài biên dọc > 1 m thường là trọng tài/khán giả
THUMB = 0.25  # tỉ lệ ảnh thu nhỏ dùng lấy màu áo


def shirt_signature(thumb, box, kps):
    """Histogram màu vùng áo (giữa vai và hông) — phân biệt hai người cùng đội khi chạy cắt nhau."""
    if thumb is None:
        return None
    sh, hp = kps[[L_SHOULDER, R_SHOULDER]], kps[[L_HIP, R_HIP]]
    if (sh[:, 2] > 0.4).all() and (hp[:, 2] > 0.4).all():
        x0, x1 = min(sh[:, 0].min(), hp[:, 0].min()), max(sh[:, 0].max(), hp[:, 0].max())
        y0, y1 = sh[:, 1].mean(), hp[:, 1].mean()
    else:
        w, h = box[2] - box[0], box[3] - box[1]
        x0, x1, y0, y1 = box[0] + .25 * w, box[2] - .25 * w, box[1] + .18 * h, box[1] + .5 * h
    x0, x1, y0, y1 = (int(v * THUMB) for v in (x0, x1, y0, y1))
    crop = thumb[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if crop.size < 12:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hist = np.zeros(16, np.float32)
    grey = (hsv[:, 1] < 40) | (hsv[:, 2] < 40)
    np.add.at(hist, (hsv[~grey, 0] / 15).astype(int) % 12, hsv[~grey, 1] / 255)
    np.add.at(hist, 12 + np.minimum(3, (hsv[grey, 2] / 64).astype(int)), 1)
    s = hist.sum()
    return hist / s if s else None


def similarity(a, b):
    return float(np.sqrt(a * b).sum())


def _export(size, imgsz):
    """Xuất YOLO11-pose (trọng số chính thức Ultralytics) sang OpenVINO một lần."""
    from ultralytics import YOLO
    import shutil
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = MODELS_DIR / f"yolo11{size}-pose_openvino_model_{imgsz}"
    if not out_dir.exists():
        out = YOLO(str(MODELS_DIR / f"yolo11{size}-pose.pt")).export(format="openvino", imgsz=imgsz, verbose=False)
        shutil.move(out, out_dir)
    return next(out_dir.glob("*.xml"))


class PoseRunner:
    """YOLO11-pose qua OpenVINO: gửi ảnh bất đồng bộ, lấy kết quả theo số khung."""

    def __init__(self, roi, size="n", imgsz=960, int8_calib_frames=None):
        import openvino as ov
        self.roi = roi
        self.imgsz = imgsz
        fp32 = _export(size, imgsz)
        int8 = fp32.with_name(fp32.stem + "_int8.xml")
        if int8_calib_frames and not int8.exists():
            import nncf
            model = ov.Core().read_model(fp32)
            data = [self._prep(f)[0] for f in int8_calib_frames]
            q = nncf.quantize(model, nncf.Dataset(data), subset_size=len(data),
                              preset=nncf.QuantizationPreset.MIXED, fast_bias_correction=True)
            ov.save_model(q, int8)
        xml = int8 if int8.exists() else fp32
        self.desc = f"YOLO11{size}-pose {imgsz} {'INT8' if xml == int8 else 'FP32'}"
        compiled = ov.Core().compile_model(str(xml), "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
        self.queue = ov.AsyncInferQueue(compiled, compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
        self.queue.set_callback(self._done)
        self.raw = {}
        self.thumbs = {}  # ảnh thu nhỏ của khung đang chờ kết quả (để lấy màu áo)

    def _prep(self, frame):
        x, y, w, h = self.roi
        crop = frame[y:y + h, x:x + w]
        r = self.imgsz / max(w, h)
        img = cv2.resize(crop, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        blob = canvas[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) * (1 / 255.0)
        return blob, r

    def _done(self, req, key):
        self.raw[key[0]] = (req.get_output_tensor(0).data[0].copy(), key[1])

    def submit(self, frame_idx, frame):
        blob, r = self._prep(frame)
        self.thumbs[frame_idx] = cv2.resize(frame, None, fx=THUMB, fy=THUMB, interpolation=cv2.INTER_AREA)
        self.queue.start_async({0: blob}, (frame_idx, r))

    def collect(self, wait=False):
        """Trả về {frame: [(box, kps, màu áo), ...]} của các khung đã xong."""
        if wait:
            self.queue.wait_all()
        done, self.raw = self.raw, {}
        out = {}
        for f, (raw, r) in done.items():
            thumb = self.thumbs.pop(f, None)
            out[f] = [(box, kps, shirt_signature(thumb, box, kps)) for box, kps in self._decode(raw, r)]
        return out

    def _decode(self, out, r, conf=0.3):
        # out: (56, A) = cx, cy, w, h, score, 17 × (x, y, v)
        keep = out[4] > conf
        if not keep.any():
            return []
        o = out[:, keep]
        cx, cy, w, h, s = o[:5]
        boxes = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), s.tolist(), conf, 0.5)
        x0, y0 = self.roi[:2]
        res = []
        for i in np.array(idx).reshape(-1):
            bx, by, bw, bh = boxes[i] / r
            box = np.array([bx + x0, by + y0, bx + bw + x0, by + bh + y0])
            kps = o[5:, i].reshape(17, 3).copy()
            kps[:, 0] = kps[:, 0] / r + x0
            kps[:, 1] = kps[:, 1] / r + y0
            res.append((box, kps))
        return res


@dataclass
class Obs:
    box: np.ndarray       # x1, y1, x2, y2 (ảnh gốc)
    kps: np.ndarray       # (17, 3) x, y, conf
    foot: np.ndarray      # điểm chạm sân (ảnh)
    pos: np.ndarray       # vị trí trên sân (m)
    sig: np.ndarray = None  # màu áo


@dataclass
class PlayerTrack:
    pid: int
    name: str
    side: int             # 0 = nửa sân gần camera, 1 = xa
    slot: int             # 0 = trái, 1 = phải (theo góc nhìn camera)
    frames: dict = field(default_factory=dict)   # frame → Obs
    last_pos: np.ndarray = None
    last_vel: np.ndarray = None
    last_frame: int = -10**9
    sig: np.ndarray = None

    def on_own_half(self, y, tol=0.6):
        return y > C.NET - tol if self.side == 0 else y < C.NET + tol


def foot_point(box, kps):
    ank = kps[[L_ANKLE, R_ANKLE]]
    good = ank[ank[:, 2] > 0.4]
    if len(good):
        return np.array([good[:, 0].mean(), good[:, 1].max()])
    return np.array([(box[0] + box[2]) / 2, box[3]])


def body_height_m(court, foot, box):
    """Chiều cao người ước tính (m): chiều cao khung chia cho tỉ lệ px/m của mặt sân tại chân, theo
    hướng vuông góc tầm nhìn (hướng ít bị co ngắn nhất ≈ cùng tỉ lệ với vật thẳng đứng)."""
    p = court.to_court([foot])[0]
    q = court.to_image([p, p + (0.25, 0), p + (0, 0.25)])
    J = np.c_[q[1] - q[0], q[2] - q[0]] / 0.25
    s = np.linalg.svd(J, compute_uv=False)[0]
    return (box[3] - box[1]) / s if s > 1e-6 else 0.0


def static_cells(pose_dets, court_at, cell=0.5, frac=0.25):
    """Ô ngoài sân có người đứng gần như suốt video (trọng tài, người xem sát biên) → bỏ qua."""
    occ, n = {}, 0
    for f in sorted(pose_dets)[::4]:
        n += 1
        ct = court_at(f)
        seen = set()
        for box, kps, _ in pose_dets[f]:
            x, y = ct.to_court([foot_point(box, kps)])[0]
            if abs(x) < 50 and abs(y) < 50:
                seen.add((int(np.floor(x / cell)), int(np.floor(y / cell))))
        for k in seen:
            occ[k] = occ.get(k, 0) + 1
    out = set()
    for (i, j) in occ:
        x, y = (i + .5) * cell, (j + .5) * cell
        inside = -0.2 < x < C.W + 0.2 and -0.2 < y < C.L + 0.2
        if inside:
            continue
        tot = sum(occ.get((i + a, j + b), 0) for a in (-1, 0, 1) for b in (-1, 0, 1))
        if n and tot / n > frac:
            out.add((i, j))
    return out, cell


class IdentityAssigner:
    """Gán người phát hiện được vào đúng người chơi theo vị trí trên sân (giữ phía sân, gần vị trí cũ)."""

    def __init__(self, court, names, singles, fps, frame_size=None, static=None):
        self.court = court
        self.fps = fps
        self.frame_size = frame_size
        self.static = static or (set(), 0.5)
        self.dropped = {"mép ảnh": 0, "chiều cao": 0, "đứng yên ngoài sân": 0, "ngoài sân": 0}
        sides = [0, 1] if singles else [0, 0, 1, 1]
        slots = [0, 0] if singles else [0, 1, 0, 1]
        self.players = [PlayerTrack(i, n, s, sl) for i, (n, s, sl) in enumerate(zip(names, sides, slots))]

    def update(self, frame_idx, dets):
        obs = []
        cells, cell = self.static
        for box, kps, sig in dets:
            foot = foot_point(box, kps)
            pos = self.court.to_court([foot])[0]
            if not (-MARGIN_X <= pos[0] <= C.W + MARGIN_X and -MARGIN_Y <= pos[1] <= C.L + MARGIN_Y):
                self.dropped["ngoài sân"] += 1
                continue
            if self.frame_size and (box[0] < 3 or box[2] > self.frame_size[0] - 3):
                # Bị mép ảnh cắt mất một phần: chỉ giữ khi thấy rõ cổ chân và đứng trong sân
                # (người đứng ngoài biên sát mép ảnh thường là trọng tài biên / người xem).
                ank_ok = (kps[[L_ANKLE, R_ANKLE], 2] > 0.5).any()
                if not (ank_ok and 0.0 <= pos[0] <= C.W and -0.5 <= pos[1] <= C.L + 1.5):
                    self.dropped["mép ảnh"] += 1
                    continue
            hm = body_height_m(self.court, foot, box)
            if not 1.0 <= hm <= 2.8:  # quá to/nhỏ so với người đứng ở chỗ đó → người ở chỗ khác hoặc nhận nhầm
                self.dropped["chiều cao"] += 1
                continue
            if (int(np.floor(pos[0] / cell)), int(np.floor(pos[1] / cell))) in cells:
                self.dropped["đứng yên ngoài sân"] += 1
                continue
            obs.append(Obs(box, kps, foot, pos, sig))
        if not obs:
            return

        P = self.players
        cost = np.full((len(P), len(obs)), 1e6)
        for i, p in enumerate(P):
            for j, o in enumerate(obs):
                if not p.on_own_half(o.pos[1]):
                    continue
                gap = (frame_idx - p.last_frame) / self.fps
                if p.last_pos is not None and gap < 3:
                    pred = p.last_pos + (p.last_vel if p.last_vel is not None else 0) * min(gap, 0.3)
                    d = np.hypot(*(o.pos - pred))
                    if d > 1.5 + 4.0 * gap:
                        continue
                    cost[i, j] = d
                    if p.sig is not None and o.sig is not None:
                        cost[i, j] += 1.5 * (1 - similarity(p.sig, o.sig))
                else:
                    # Chưa thấy / mất dấu lâu: ưu tiên đúng bên trái/phải của mình.
                    teammates = sum(q.side == p.side for q in P)
                    frac = np.clip(o.pos[0] / C.W, 0, 1)
                    cost[i, j] = 2.0 + (2.0 * (frac if p.slot == 0 else 1 - frac) if teammates == 2 else 0)

        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] >= 1e5:
                continue
            p, o = P[i], obs[j]
            if p.last_pos is not None:
                dt = max(1, frame_idx - p.last_frame) / self.fps
                v = (o.pos - p.last_pos) / dt
                p.last_vel = v if p.last_vel is None else 0.6 * p.last_vel + 0.4 * v
            if o.sig is not None:
                p.sig = o.sig.copy() if p.sig is None else 0.9 * p.sig + 0.1 * o.sig
            p.frames[frame_idx] = o
            p.last_pos = o.pos
            p.last_frame = frame_idx
