"""Kích thước sân và phép chiếu phối cảnh ảnh ↔ mặt sân (mét)."""
import json
from pathlib import Path

import cv2
import numpy as np

# Gốc toạ độ: góc trái biên cuối phía XA camera. x: ngang sân, y: dọc sân (0 = xa, 13.4 = gần).
W, L, NET = 6.1, 13.4, 6.7
SHORT, LONG_D, SINGLE_IN = 1.98, 0.76, 0.46
CORNERS_M = np.float32([[0, 0], [W, 0], [W, L], [0, L]])


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
    def __init__(self, corners_px):
        self.corners = np.float32(corners_px)
        self.H = cv2.getPerspectiveTransform(self.corners, CORNERS_M)        # ảnh → sân
        self.Hinv = cv2.getPerspectiveTransform(CORNERS_M, self.corners)     # sân → ảnh

    def to_court(self, pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.Hinv).reshape(-1, 2)

    def px_per_meter(self, court_xy):
        """Số pixel ứng với 1 m theo chiều ngang sân tại một điểm trên sân (để quy đổi tốc độ trên ảnh)."""
        x, y = court_xy
        a, b = self.to_image([[x - 0.5, y], [x + 0.5, y]])
        return float(np.hypot(*(b - a)))

    def roi(self, frame_w, frame_h, margin_m=(1.6, 2.6), head_m=2.4):
        """Vùng ảnh quanh sân (kèm khoảng ngoài sân, chiều cao người phía xa)."""
        mx, my = margin_m
        pts = self.to_image([[-mx, -my], [W + mx, -my], [W + mx, L + my], [-mx, L + my]])
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        far_w = np.hypot(*(self.corners[1] - self.corners[0]))
        y0 -= far_w / W * head_m
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(frame_w, int(x1)), min(frame_h, int(y1))
        return x0, y0, x1 - x0, y1 - y0

    def save(self, path, frame_w, frame_h):
        data = {"corners": [[float(x) / frame_w, float(y) / frame_h] for x, y in self.corners]}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def load(path, frame_w, frame_h):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Court([[x * frame_w, y * frame_h] for x, y in data["corners"]])
