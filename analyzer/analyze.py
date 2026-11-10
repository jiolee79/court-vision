"""Court Vision PC — phân tích video cầu lông: theo dõi người + quả cầu, xuất video HUD và Excel.

    python analyze.py VIDEO [--names "A,B,C,D"] [--singles] [--start 0] [--end 0] [--quality fast|best]
"""
import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from calibrate import calibrate  # noqa: E402
from court import MovingCourt, load_calibration, save_calibration  # noqa: E402
from events import (audio_onsets, build_player_series, build_rallies, clean_shuttle,  # noqa: E402
                    detect_hits, estimate_speeds)
from players import IdentityAssigner, PoseRunner, static_cells  # noqa: E402
from render import open_video, render_video, thumb  # noqa: E402
from report import write_excel  # noqa: E402
from scenes import camera_segments, track_camera  # noqa: E402
from shuttle import (SKY_M, ShuttleTracker, TrackNetRunner, calibration_samples, ensure_int8,  # noqa: E402
                     median_background, sample_frames, segment_frames, tiles_for_rect)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_audio(path, start, dur, sr=22050):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}"] + (["-t", f"{dur:.3f}"] if dur else []) + \
          ["-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return (np.frombuffer(raw, np.float32) if raw else None), sr


def writable(path):
    """File kết quả đang mở trong Excel / trình xem video thì không ghi đè được → đổi sang tên khác
    thay vì báo lỗi sau khi đã phân tích xong."""
    try:
        with open(path, "ab"):
            pass
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{time.strftime('%H%M%S')}{path.suffix}")
        log(f"File {path.name} đang được mở ở chương trình khác → lưu thành {alt.name}")
        return alt


def rally_mask(onsets, n, fps, before=2.0, after=3.0, max_gap=4.0):
    """Khung nào có thể đang trong rally: quanh các chuỗi tiếng vợt cách nhau không quá max_gap giây."""
    mask = np.zeros(n, bool)
    if len(onsets) < 3:
        mask[:] = True  # không nghe rõ (video không tiếng / quá ít) → quét toàn bộ cho chắc
        return mask
    groups, cur = [], [onsets[0]]
    for t in onsets[1:]:
        if t - cur[-1] <= max_gap:
            cur.append(t)
        else:
            groups.append(cur)
            cur = [t]
    groups.append(cur)
    for g in groups:
        a, b = int((g[0] - before) * fps), int((g[-1] + after) * fps)
        mask[max(0, a):min(n, b)] = True
    return mask


def frames(cap, start_frame, end_frame, step):
    """Đọc khung tuần tự trên luồng riêng (giải mã song song với AI), bỏ bớt khung khi video 60 fps."""
    import queue
    import threading
    q = queue.Queue(maxsize=48)

    def reader():
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        i = start_frame
        while i < end_frame:
            if (i - start_frame) % step:
                if not cap.grab():
                    break
            else:
                ok, f = cap.read()
                if not ok:
                    break
                q.put(f)
            i += 1
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    while (f := q.get()) is not None:
        yield f


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, không in được tiếng Việt
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--names", default="", help="Tên người chơi, cách nhau dấu phẩy: gần-trái, gần-phải, xa-trái, xa-phải")
    ap.add_argument("--singles", action="store_true", help="Đánh đơn (2 người)")
    ap.add_argument("--start", type=float, default=0, help="Bắt đầu từ giây thứ")
    ap.add_argument("--end", type=float, default=0, help="Kết thúc ở giây thứ (0 = hết video)")
    ap.add_argument("--quality", choices=["fast", "best"], default="fast")
    ap.add_argument("--recalibrate", action="store_true", help="Hiệu chỉnh lại sân dù đã có file lưu")
    ap.add_argument("--out", default="", help="Thư mục kết quả (mặc định: cạnh video)")
    args = ap.parse_args()

    video = Path(args.video).resolve()
    cap = open_video(video)
    if not cap.isOpened():
        sys.exit(f"Không mở được video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = 2 if fps > 40 else 1               # TrackNet học ở ~30 fps
    afps = fps / step
    start_f = int(args.start * fps)
    end_f = min(total, int(args.end * fps)) if args.end else total
    n = len(range(start_f, end_f, step))
    log(f"Video {W}x{H}, {fps:.1f} fps, {total / fps / 60:.1f} phút → phân tích {n} khung ở {afps:.0f} fps")

    out_dir = Path(args.out) if args.out else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem + (f"_{int(args.start)}-{int(args.end)}s" if args.start or args.end else "")

    # --- Hiệu chỉnh sân (lưu cạnh video để lần sau dùng lại)
    court_file = video.with_suffix(".court.json")
    dur_s = (end_f - start_f) / fps

    def frame_at(t):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        return f if ok else np.zeros((H, W, 3), np.uint8)

    if court_file.exists() and not args.recalibrate:
        ref_court, ref_t = load_calibration(court_file, W, H)
        log(f"Dùng hiệu chỉnh sân đã lưu: {court_file.name}")
    else:
        res = calibrate(frame_at, args.start + min(5.0, dur_s / 2), total / fps)
        if res is None:
            sys.exit("Đã huỷ hiệu chỉnh sân.")
        ref_court, ref_t = res
        save_calibration(court_file, ref_court, ref_t, W, H)

    # Giữ đúng vị trí từng ô tên: ô bỏ trống dùng tên mặc định (không dồn tên người sau lên trước).
    defaults = ["Người 1", "Người 3"] if args.singles else ["Người 1", "Người 2", "Người 3", "Người 4"]
    given = [s.strip() for s in args.names.split(",")] if args.names.strip() else []
    names = [(given[i] if i < len(given) and given[i] else d) for i, d in enumerate(defaults)]

    # --- Máy quay có bị xê dịch không? Có thì bám theo chuyển động máy để dời sân theo từng khung.
    t0 = time.time()
    seg_starts = camera_segments(video, W, H, start_s=args.start, dur_s=dur_s)
    if len(seg_starts) > 1:
        log(f"Máy quay xê dịch {len(seg_starts) - 1} lần → đo chuyển động máy để dời sân theo…")
        times, Gs, conf = track_camera(video, W, H, ref_t, args.start, dur_s)
        courts = MovingCourt(ref_court, times, Gs, conf, afps)
        bad = float(np.mean(np.asarray(conf) < 0.15))
        if bad > 0.05:
            log(f"Lưu ý: {bad * 100:.0f}% thời lượng không bám được máy quay (rung mạnh/bị che) — số liệu đoạn đó kém chính xác")
    else:
        courts = MovingCourt.fixed(ref_court, afps)

    # Vùng quét: gộp vùng sân ở mọi vị trí máy đã đứng.
    def union(rs):
        x0, y0 = min(r[0] for r in rs), min(r[1] for r in rs)
        return x0, y0, max(r[0] + r[2] for r in rs) - x0, max(r[1] + r[3] for r in rs) - y0

    sample_idx = range(0, n, max(1, int(afps * 2)))
    tiles = tiles_for_rect(union([courts.at(a).roi(W, H, margin_m=(1.0, 2.0), head_m=SKY_M) for a in sample_idx]), W, H)
    roi = union([courts.at(a).roi(W, H) for a in sample_idx])
    log(f"Chuẩn bị AI ({len(tiles)} ô quét cầu)…")
    samples = sample_frames(video, W, H, total / fps)
    # Ảnh nền riêng cho từng đoạn máy đứng yên một chỗ.
    seg_bounds = [int(s * afps) for s in seg_starts] + [n]
    seg_medians = []
    for k in range(len(seg_starts)):
        seg_len = max(1.0, (seg_bounds[k + 1] - seg_bounds[k]) / afps)
        fr = segment_frames(video, W, H, args.start + seg_starts[k], seg_len) or samples
        seg_medians.append(median_background(fr, tiles))
    medians = seg_medians[0]
    ensure_int8(lambda: calibration_samples(video, tiles, medians))
    tracknet = TrackNetRunner(int8=True)
    pose = PoseRunner(roi, size="s" if args.quality == "best" else "n", imgsz=960, int8_calib_frames=samples[:40])
    log(f"Cầu: TrackNetV3 {tracknet.desc} · Người: {pose.desc}")

    pcm, sr = read_audio(video, args.start, dur_s)
    onsets = audio_onsets(pcm, sr)

    # --- Lượt 1: nhận diện (lưu lại để lần sau đổi tên / xuất lại video không phải chạy AI lại)
    # Phân tích một đoạn dùng file lưu riêng, không ghi đè kết quả AI của cả video.
    cache_file = video.with_name(stem + ".cvcache.pkl")
    cache_key = ("v2", start_f, end_f, step, args.quality, ref_court.ref, tuple(map(tuple, np.round(ref_court.corners, 1))),
                 round(ref_t, 2), len(seg_starts), tuple(tiles))
    cached = None
    if cache_file.exists():
        try:
            cached = pickle.loads(cache_file.read_bytes())
            if cached.get("key") != cache_key:
                cached = None
        except Exception:  # noqa: BLE001 — file hỏng thì chạy lại
            cached = None
    if cached:
        raw_shuttle, pose_dets, thumbs = cached["shuttle"], cached["pose"], cached["thumbs"]
        log("Dùng kết quả nhận diện đã lưu (bỏ qua bước AI)")
    else:
        # Nghe trước: đoạn không có tiếng vợt > 4 giây là lúc nghỉ → không cần quét quả cầu.
        active = rally_mask(onsets, n, afps)
        if active.mean() < 0.85:
            log(f"Bỏ qua quét cầu ở {100 * (1 - active.mean()):.0f}% thời lượng (đoạn nghỉ giữa các rally)")
        else:
            active[:] = True  # sân ồn hoặc chơi liên tục: quét toàn bộ
        shuttle = ShuttleTracker(tracknet, tiles, medians)
        pose_every = 2 if args.quality == "fast" else 1
        pose_dets = {}
        thumbs = []
        seg = 0
        for a, frame in enumerate(tqdm(frames(cap, start_f, end_f, step), total=n, desc="Nhận diện", unit="khung")):
            while seg + 1 < len(seg_starts) and a >= seg_bounds[seg + 1]:
                seg += 1
                shuttle.set_medians(seg_medians[seg])  # máy sang vị trí mới → ảnh nền mới
            if active[min(a, n - 1)]:
                shuttle.push(a, frame)
            else:
                shuttle.gap()
            thumbs.append(thumb(frame))
            if a % (pose_every if active[min(a, n - 1)] else 6) == 0:  # lúc nghỉ: theo dõi người thưa hơn
                pose.submit(a, frame)
            if a % 64 == 0:
                pose_dets.update(pose.collect())
        raw_shuttle = shuttle.flush()
        pose_dets.update(pose.collect(wait=True))
        thumbs = np.array(thumbs)
        cache_file.write_bytes(pickle.dumps({"key": cache_key, "shuttle": raw_shuttle, "pose": pose_dets,
                                             "thumbs": thumbs}))
        log(f"Nhận diện xong sau {time.time() - t0:.0f}s")
    n = max(len(thumbs), max(raw_shuttle, default=0) + 1)

    # --- Ghép người, cú đánh, rally
    ia = IdentityAssigner(courts.at(0), names, args.singles, afps, frame_size=(W, H),
                          static=static_cells(pose_dets, courts.at))
    for a in sorted(pose_dets):
        ia.court = courts.at(a)
        ia.update(a, pose_dets[a])
    d = ia.dropped
    log(f"Lọc bỏ người không phải VĐV: {d['ngoài sân']} ngoài sân, {d['mép ảnh']} bị mép ảnh cắt, "
        f"{d['chiều cao']} sai kích thước, {d['đứng yên ngoài sân']} đứng yên cạnh sân")
    players = [build_player_series(t, n, afps) for t in ia.players]
    shuttle_xy = clean_shuttle(raw_shuttle, n, H)
    hits = detect_hits(shuttle_xy, players, afps, onsets)
    estimate_speeds(hits, shuttle_xy, players, courts, afps)
    rallies = build_rallies(hits, shuttle_xy, courts, afps, players=players)
    hits = [h for r in rallies for h in r.hits]  # bỏ cú lẻ ngoài rally (nhiễu)
    for p in players:
        p.hits = [h for h in hits if h.player == p.pid]
    log(f"Cầu thấy ở {np.mean(~np.isnan(shuttle_xy[:, 0])) * 100:.0f}% khung · {len(hits)} cú đánh · {len(rallies)} rally")

    # --- Excel
    xlsx = writable(out_dir / f"{stem}_thong-ke.xlsx")
    overview, summary = write_excel(xlsx, players, rallies, hits, afps, args.start)

    # --- Lượt 2: xuất video HUD
    cap.release()
    out_mp4 = writable(out_dir / f"{stem}_HUD.mp4")
    workers = max(1, min(8, (os.cpu_count() or 4) // 3))
    log(f"Xuất video HUD trên {workers} tiến trình song song…")
    t1 = time.time()
    with tqdm(total=n, desc="Xuất video", unit="khung") as bar:
        render_video(out_mp4, video, start_f, step, n, (W, H, players, rallies, hits, shuttle_xy, afps), thumbs,
                     args.start, (end_f - start_f) / fps, workers, on_progress=bar.update)
    log(f"Xuất video xong sau {time.time() - t1:.0f}s")

    log(f"Hoàn tất sau {(time.time() - t0) / 60:.1f} phút")
    print("\n" + overview.to_string(index=False))
    print("\n" + summary.to_string(index=False))
    print(f"\nVideo: {out_mp4}\nExcel: {xlsx}")


if __name__ == "__main__":
    main()
