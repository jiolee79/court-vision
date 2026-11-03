"""Nhận diện quả cầu bằng TrackNetV3 (mô hình chuyên cho cầu lông, MIT License).

TrackNet nhận chuỗi 8 khung hình 512x288 (16:9) + ảnh nền trung vị và trả về heatmap vị trí cầu.
Video quay dọc bằng điện thoại không phải 16:9, nên vùng sân + khoảng không phía trên được cắt
thành các ô 16:9 xếp chồng (tile) để quả cầu không bị bóp nhỏ.

Tăng tốc trên CPU: OpenVINO + nén INT8 (NNCF) + chạy song song nhiều ô/nhiều cửa sổ cùng lúc.
"""
import cv2
import numpy as np

from paths import TRACKNET_DIR

HEIGHT, WIDTH = 288, 512
SEQ = 8
FP32_XML = TRACKNET_DIR / "tracknet_ov.xml"
INT8_XML = TRACKNET_DIR / "tracknet_int8.xml"


def _export_fp32():
    import torch
    import openvino as ov
    from tracknet_model import TrackNet

    # weights_only=True: chỉ nạp dữ liệu số, không thực thi mã nào trong file checkpoint.
    ckpt = torch.load(TRACKNET_DIR / "TrackNet_best.pt", map_location="cpu", weights_only=True)
    assert ckpt["param_dict"]["seq_len"] == SEQ and ckpt["param_dict"]["bg_mode"] == "concat"
    net = TrackNet(in_dim=(SEQ + 1) * 3, out_dim=SEQ)
    net.load_state_dict(ckpt["model"])
    net.eval()
    model = ov.convert_model(net, example_input=torch.zeros(1, (SEQ + 1) * 3, HEIGHT, WIDTH))
    ov.save_model(model, FP32_XML)


def ensure_int8(get_calib_samples):
    """Nén INT8 một lần (dùng vài chục mẫu thật để hiệu chỉnh), các lần sau dùng lại."""
    if INT8_XML.exists():
        return
    import nncf
    import openvino as ov
    if not FP32_XML.exists():
        _export_fp32()
    model = ov.Core().read_model(FP32_XML)
    calib_samples = get_calib_samples()
    q = nncf.quantize(model, nncf.Dataset(calib_samples), subset_size=len(calib_samples),
                      preset=nncf.QuantizationPreset.MIXED, fast_bias_correction=True)
    ov.save_model(q, INT8_XML)


class TrackNetRunner:
    def __init__(self, int8=True):
        import openvino as ov
        if not FP32_XML.exists():
            _export_fp32()
        xml = INT8_XML if int8 and INT8_XML.exists() else FP32_XML
        self.desc = "INT8" if xml == INT8_XML else "FP32"
        core = ov.Core()
        self.compiled = core.compile_model(str(xml), "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
        self.n = self.compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        self.queue = ov.AsyncInferQueue(self.compiled, self.n)
        self.out = {}
        self.queue.set_callback(self._done)

    def _done(self, req, key):
        self.out[key] = req.get_output_tensor(0).data[0].copy()

    def run_many(self, items):
        """items: [(key, input)] → {key: heatmaps (8, 288, 512)}, chạy song song."""
        self.out = {}
        for key, x in items:
            self.queue.start_async({0: x}, key)
        self.queue.wait_all()
        return self.out


def plan_tiles(court, frame_w, frame_h):
    """Các ô 16:9 phủ vùng sân và khoảng không phía trên (cầu bay cao)."""
    x, y, w, h = court.roi(frame_w, frame_h, margin_m=(1.0, 2.0), head_m=4.0)
    tile_h = int(round(w * 9 / 16))
    if tile_h >= h:
        # Video ngang: một ô, mở rộng theo chiều dọc cho đủ 16:9.
        th = min(frame_h, tile_h)
        ty = int(np.clip(y + h / 2 - th / 2, 0, frame_h - th))
        tw = min(frame_w, int(round(th * 16 / 9)))
        tx = int(np.clip(x + w / 2 - tw / 2, 0, frame_w - tw))
        return [(tx, ty, tw, th)]
    n = int(np.ceil((h - tile_h) / (tile_h * 0.9))) + 1
    step = (h - tile_h) / max(1, n - 1)
    return [(x, int(y + i * step), w, tile_h) for i in range(n)]


def _prep(frame, tile):
    x, y, w, h = tile
    img = cv2.resize(frame[y:y + h, x:x + w], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))  # BGR→RGB, HWC→CHW, uint8


def sample_frames(video_path, frame_w, frame_h, duration, max_frames=150):
    """Lấy nhanh các khung chính (keyframe) bằng ffmpeg — không phải giải mã cả video."""
    import subprocess
    size = frame_w * frame_h * 3

    def run(args, keep_every):
        cmd = ["ffmpeg", "-v", "error"] + args + ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        out = []
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
            k = 0
            while True:
                buf = p.stdout.read(size)
                if len(buf) < size:
                    break
                if k % keep_every == 0:
                    out.append(np.frombuffer(buf, np.uint8).reshape(frame_h, frame_w, 3))
                k += 1
        return out

    # Điện thoại thường có ~1 keyframe/giây → chỉ giải mã keyframe, rất nhanh.
    frames = run(["-skip_frame", "nokey", "-i", str(video_path), "-vsync", "vfr"],
                 max(1, int(duration) // max_frames))
    if len(frames) < 10:
        # Video ít keyframe (thường là clip ngắn đã qua mạng xã hội): lấy đều ~40 khung.
        frames = run(["-i", str(video_path), "-vf", f"fps={min(4.0, 40 / max(duration, 1)):.4f}"], 1)
    return frames[::max(1, len(frames) // max_frames)]


def median_background(frames, tiles):
    return [np.median(np.stack([_prep(f, t) for f in frames]), 0).astype(np.uint8) for t in tiles]


def calibration_samples(video_path, tiles, medians, n=30):
    """n cửa sổ 8 khung liên tiếp từ đầu video để hiệu chỉnh INT8 (chỉ cần một lần)."""
    cap = cv2.VideoCapture(str(video_path))
    out = []
    for k in range(n):
        frames = [cap.read()[1] for _ in range(SEQ)]
        for _ in range(SEQ):  # bỏ qua một đoạn để mẫu đa dạng hơn
            cap.grab()
        if any(f is None for f in frames):
            break
        t = k % len(tiles)
        stack = [medians[t]] + [_prep(f, tiles[t]) for f in frames]
        out.append((np.concatenate(stack, 0)[None].astype(np.float32) / 255.0))
    cap.release()
    return out


class ShuttleTracker:
    """Nhận từng khung hình theo thứ tự; gom nhiều cửa sổ 8 khung rồi chạy song song."""

    def __init__(self, runner, tiles, medians, batch_windows=4):
        self.runner = runner
        self.tiles = tiles
        self.medians = medians
        self.batch_windows = batch_windows
        self.window = []      # [(frame_idx, [tile imgs])]
        self.pending = []     # các cửa sổ đủ 8 khung chờ chạy
        self.results = {}
        self.conf = {}

    def push(self, frame_idx, frame):
        self.window.append((frame_idx, [_prep(frame, t) for t in self.tiles]))
        if len(self.window) == SEQ:
            self.pending.append(self.window)
            self.window = []
            if len(self.pending) >= self.batch_windows:
                self._run()

    def gap(self):
        """Báo chuỗi khung bị ngắt (đoạn nghỉ được bỏ qua): khép cửa sổ đang dở bằng cách lặp khung cuối."""
        if self.window:
            last = self.window[-1]
            while len(self.window) < SEQ:
                self.window.append((None, last[1]))
            self.pending.append(self.window)
            self.window = []
            if len(self.pending) >= self.batch_windows:
                self._run()

    def flush(self):
        self.gap()
        if self.pending:
            self._run()
        return self.results

    def _run(self):
        items = []
        for w, win in enumerate(self.pending):
            for t in range(len(self.tiles)):
                stack = [self.medians[t]] + [imgs[t] for _, imgs in win]
                items.append(((w, t), np.concatenate(stack, 0)[None].astype(np.float32) * (1 / 255.0)))
        heats = self.runner.run_many(items)
        for w, win in enumerate(self.pending):
            for k, (f, _) in enumerate(win):
                if f is None:
                    continue
                self.results[f], self.conf[f] = self._locate([heats[(w, t)][k] for t in range(len(self.tiles))])
        self.pending = []

    def _locate(self, hms):
        best = (0.0, None)
        for t, hm in enumerate(hms):
            peak = float(hm.max())
            if peak < 0.5 or peak <= best[0]:
                continue
            mask = (hm > 0.5).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            tx, ty, tw, th = self.tiles[t]
            best = (peak, (tx + (bx + bw / 2) * tw / WIDTH, ty + (by + bh / 2) * th / HEIGHT))
        return best[1], best[0]
