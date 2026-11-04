"""Phát hiện máy quay bị xê dịch giữa chừng → chia video thành các đoạn, mỗi đoạn hiệu chỉnh sân riêng.

So sánh nửa trên khung hình (trần, tường — đứng yên, ít người qua lại) bằng tương quan pha (phase
correlation): người chạy bên dưới không ảnh hưởng, còn máy dịch vài pixel là nhận ra ngay.
"""
import subprocess

import cv2
import numpy as np

SAMPLE_FPS = 2
SMALL_W = 160


TRACK_FPS = 5
TRACK_W = 480


def _read_small(video, frame_w, frame_h, fps, width, start_s=0.0, dur_s=None):
    sh = int(round(frame_h * width / frame_w))
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start_s:.3f}"] + (["-t", f"{dur_s:.3f}"] if dur_s else []) + \
          ["-i", str(video), "-vf", f"fps={fps},scale={width}:{sh},format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    size = width * sh
    return [np.frombuffer(raw[i:i + size], np.uint8).reshape(sh, width) for i in range(0, len(raw) - size + 1, size)]


def track_camera(video, frame_w, frame_h, ref_s, start_s=0.0, dur_s=None, on_progress=None):
    """Bám chuyển động máy quay: với mỗi mẫu (5 lần/giây) trả về homography G sao cho
    điểm_ảnh_tại_mẫu = G · điểm_ảnh_khung_tham_chiếu.

    Dùng điểm đặc trưng ORB ở nửa trên khung hình (trần, tường — cố định). Khi máy chỉ xoay quanh chỗ đặt,
    phép biến đổi này đúng cho mọi điểm trong hình, kể cả mặt sân.
    Trả về (thời điểm các mẫu tính từ start_s, danh sách G 3x3, danh sách độ tin cậy 0..1).
    """
    small = _read_small(video, frame_w, frame_h, TRACK_FPS, TRACK_W, start_s, dur_s)
    if not small:
        return np.array([0.0]), [np.eye(3)], [1.0]
    sh = small[0].shape[0]
    mask = np.zeros_like(small[0])
    mask[: int(sh * 0.55)] = 255
    orb = cv2.ORB_create(1500, fastThreshold=12)
    feats = [orb.detectAndCompute(f, mask) for f in small]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(i, j):
        (ka, da), (kb, db) = feats[i], feats[j]
        if da is None or db is None or len(ka) < 30 or len(kb) < 30:
            return None, 0
        m = bf.match(da, db)
        if len(m) < 30:
            return None, 0
        a = np.float32([ka[x.queryIdx].pt for x in m])
        b = np.float32([kb[x.trainIdx].pt for x in m])
        G, inl = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
        n = int(inl.sum()) if inl is not None else 0
        return (G, n) if G is not None and n >= 30 else (None, n)

    ref = int(np.clip(round((ref_s - start_s) * TRACK_FPS), 0, len(small) - 1))
    G = [None] * len(small)
    conf = [0.0] * len(small)
    G[ref], conf[ref] = np.eye(3), 1.0
    done = 0
    for order in (range(ref + 1, len(small)), range(ref - 1, -1, -1)):
        prev = ref
        for i in order:
            g, n = match(ref, i)                       # so thẳng với khung tham chiếu (không cộng dồn sai số)
            if g is None:
                m, n = match(prev, i)                  # không được thì nối tiếp từ mẫu trước
                g = m @ G[prev] if m is not None else G[prev]
                n = n if m is not None else 0
            G[i], conf[i] = g, min(1.0, n / 150)
            prev = i
            done += 1
            if on_progress and done % 50 == 0:
                on_progress(done / len(small))
    # Đổi từ toạ độ ảnh nhỏ sang ảnh gốc.
    s = frame_w / TRACK_W
    S, Si = np.diag([s, s, 1.0]), np.diag([1 / s, 1 / s, 1.0])
    times = np.arange(len(small)) / TRACK_FPS
    return times, [S @ g @ Si for g in G], conf


def camera_segments(video, frame_w, frame_h, shift_px=4.0, min_len_s=6.0, start_s=0.0, dur_s=None):
    """Trả về danh sách thời điểm (giây, tính từ start_s) bắt đầu mỗi đoạn máy đứng yên."""
    sh = int(round(frame_h * SMALL_W / frame_w))
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start_s:.3f}"] + (["-t", f"{dur_s:.3f}"] if dur_s else []) + \
          ["-i", str(video), "-vf", f"fps={SAMPLE_FPS},scale={SMALL_W}:{sh},format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    size = SMALL_W * sh
    frames = [np.frombuffer(raw[i:i + size], np.uint8).reshape(sh, SMALL_W)[: int(sh * 0.45)].astype(np.float32)
              for i in range(0, len(raw) - size + 1, size)]
    if len(frames) < 4:
        return [0.0]
    win = cv2.createHanningWindow(frames[0].shape[::-1], cv2.CV_32F)

    starts, ref, moved = [0.0], frames[0], 0
    for i, f in enumerate(frames[1:], 1):
        (dx, dy), resp = cv2.phaseCorrelate(ref, f, win)
        if np.hypot(dx, dy) > shift_px or resp < 0.08:
            moved += 1
            if moved >= 2:  # lệch ổn định 2 mẫu liên tiếp → máy đã sang vị trí mới
                starts.append((i - 1) / SAMPLE_FPS)
                ref, moved = f, 0
        else:
            moved = 0
            ref = 0.9 * ref + 0.1 * f  # cập nhật dần (ánh sáng thay đổi nhẹ)

    # Đoạn quá ngắn (đang cầm máy di chuyển) gộp vào đoạn kế tiếp.
    merged = [starts[0]]
    for s in starts[1:]:
        if s - merged[-1] < min_len_s and len(merged) > 1:
            merged[-1] = s
        elif s - merged[-1] < min_len_s:
            continue
        else:
            merged.append(s)
    return merged
