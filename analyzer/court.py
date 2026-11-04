"""Kích thước sân và phép chiếu phối cảnh ảnh ↔ mặt sân (mét)."""
import json
from pathlib import Path

import cv2
import numpy as np

# Gốc toạ độ: góc trái biên cuối phía XA camera. x: ngang sân, y: dọc sân (0 = xa, 13.4 = gần).
W, L, NET = 6.1, 13.4, 6.7
SHORT, LONG_D, SINGLE_IN = 1.98, 0.76, 0.46

# Bộ 4 điểm mốc dùng để hiệu chỉnh — chọn bộ nhìn thấy rõ nhất trong video.
REFS = {
    "corners": {
        "name": "4 góc sân",
        "points": [[0, 0], [W, 0], [W, L], [0, L]],
        "labels": ["Góc xa-trái", "Góc xa-phải", "Góc gần-phải", "Góc gần-trái"],
    },
    "net_short": {
        "name": "Chân 2 cột lưới + vạch giao cầu ngắn phía gần",
        "points": [[0, NET], [W, NET], [W, NET + SHORT], [0, NET + SHORT]],
        "labels": ["Chân cột lưới trái", "Chân cột lưới phải", "Vạch ngắn gần-phải", "Vạch ngắn gần-trái"],
    },
    "net_base": {
        "name": "Chân 2 cột lưới + 2 góc biên cuối phía gần",
        "points": [[0, NET], [W, NET], [W, L], [0, L]],
        "labels": ["Chân cột lưới trái", "Chân cột lưới phải", "Góc gần-phải", "Góc gần-trái"],
    },
}


def court_lines():
    s = [
        (0, 0, W, 0), (W, 0, W, L), (W, L, 0, L), (0, L, 0, 0),
        (SINGLE_IN, 0, SINGLE_IN, L), (W - SINGLE_IN, 0, W - SINGLE_IN, L),
        (0, NET - SHORT, W, NET - SHORT), (0, NET + SHORT, W, NET + SHORT),
        (0, LONG_D, W, LONG_D), (0, L - LONG_D, W, L - LONG_D),
        (W / 2, 0, W / 2, NET - SHORT), (W / 2, NET + SHORT, W / 2, L),
    ]
    return s


class Court:
    def __init__(self, points_px, ref="corners"):
        self.ref = ref
        self.corners = np.float32(points_px)   # 4 điểm mốc trên ảnh (theo bộ mốc `ref`)
        dst = np.float32(REFS[ref]["points"])
        self.H = cv2.getPerspectiveTransform(self.corners, dst)        # ảnh → sân
        self.Hinv = cv2.getPerspectiveTransform(dst, self.corners)     # sân → ảnh

    def to_court(self, pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.Hinv).reshape(-1, 2)

    def visible_image(self, pts):
        """Chiếu điểm sân lên ảnh, bỏ những điểm nằm sau lưng camera (máy đặt thấp/sát sân)."""
        p = np.c_[np.asarray(pts, np.float64), np.ones(len(pts))] @ self.Hinv.T
        ok = p[:, 2] > 1e-6
        return p[ok, :2] / p[ok, 2:3]

    def px_per_meter(self, court_xy):
        """Số pixel ứng với 1 m theo chiều ngang sân tại một điểm trên sân (để quy đổi tốc độ trên ảnh)."""
        x, y = court_xy
        a, b = self.to_image([[x - 0.5, y], [x + 0.5, y]])
        return float(np.hypot(*(b - a)))

    def roi(self, frame_w, frame_h, margin_m=(1.2, 2.0), head_m=2.4):
        """Vùng ảnh quanh sân (kèm khoảng ngoài sân và khoảng không phía trên đến độ cao head_m)."""
        mx, my = margin_m
        gx, gy = np.meshgrid(np.linspace(-mx, W + mx, 9), np.linspace(-my, L + my, 15))
        pts = self.visible_image(np.c_[gx.ravel(), gy.ravel()])
        big = 4 * max(frame_w, frame_h)
        pts = pts[(np.abs(pts) < big).all(1)]
        if not len(pts):
            return 0, 0, frame_w, frame_h
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        # Chiều cao head_m ở biên cuối xa ~ head_m × (số pixel / mét theo chiều ngang tại đó).
        far = self.visible_image([[0, 0], [W, 0]])
        ppm = np.hypot(*(far[1] - far[0])) / W if len(far) == 2 else 40
        y0 -= ppm * head_m
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(frame_w, int(x1)), min(frame_h, int(y1))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return 0, 0, frame_w, frame_h
        return x0, y0, x1 - x0, y1 - y0

    def as_dict(self, frame_w, frame_h):
        return {"ref": self.ref, "points": [[float(x) / frame_w, float(y) / frame_h] for x, y in self.corners]}

    @staticmethod
    def from_dict(d, frame_w, frame_h):
        pts = d.get("points", d.get("corners"))
        return Court([[x * frame_w, y * frame_h] for x, y in pts], d.get("ref", "corners"))


class MovingCourt:
    """Sân theo từng khung khi máy quay xê dịch: hiệu chỉnh một lần ở khung tham chiếu, rồi dời các điểm
    mốc theo chuyển động máy đã đo (scenes.track_camera)."""

    def __init__(self, ref_court, times, Gs, conf, fps, sample_fps=5):
        self.ref = ref_court
        self.times, self.Gs, self.conf = np.asarray(times), Gs, np.asarray(conf)
        self.fps = fps                  # fps của các khung phân tích
        self.sample_fps = sample_fps    # tần suất đo chuyển động máy
        self._cache = {}

    @staticmethod
    def fixed(court, fps=30):
        return MovingCourt(court, [0.0], [np.eye(3)], [1.0], fps)

    def index(self, frame):
        return int(np.clip(round(frame / self.fps * self.sample_fps), 0, len(self.times) - 1))

    def at(self, frame):
        k = self.index(frame) if len(self.times) > 1 else 0
        c = self._cache.get(k)
        if c is None:
            pts = cv2.perspectiveTransform(self.ref.corners.reshape(-1, 1, 2), self.Gs[k]).reshape(-1, 2)
            c = self._cache[k] = Court(pts, self.ref.ref)
        return c

    def reliable(self, frame, min_conf=0.15):
        return len(self.times) == 1 or self.conf[self.index(frame)] >= min_conf


def save_calibration(path, court, ref_s, frame_w, frame_h):
    data = {"ref_s": round(float(ref_s), 3), **court.as_dict(frame_w, frame_h)}
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_calibration(path, frame_w, frame_h):
    """Trả về (Court ở khung tham chiếu, thời điểm khung tham chiếu)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return Court.from_dict(d, frame_w, frame_h), float(d.get("ref_s", 0.0))
