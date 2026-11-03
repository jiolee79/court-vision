"""Vẽ HUD kiểu truyền hình lên video và ghi ra MP4 (kèm âm thanh gốc)."""
import subprocess
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import court as C
from paths import FONT_BOLD, FONT_REGULAR

COLORS = [(79, 195, 247), (126, 224, 129), (199, 146, 234), (255, 183, 77)]  # RGB
PANEL = (8, 18, 28, 190)
EDGE = (120, 200, 255, 90)
ACCENT = (63, 208, 255)
LABEL = (200, 225, 245, 200)
WHITE = (255, 255, 255, 255)


@lru_cache(maxsize=64)
def font(size, bold=True):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), max(6, int(size)))


def fmt_clock(sec):
    m = int(sec // 60)
    return f"{m:02d}:{sec - m * 60:05.2f}"


def blend(frame, rgba, x, y):
    """Dán ảnh RGBA lên khung BGR tại (x, y), tự cắt phần tràn mép."""
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    # Tính bằng số nguyên 16-bit (nhanh hơn số thực ~4 lần); >> 8 xấp xỉ / 255.
    a = src[..., 3:4].astype(np.uint16)
    dst = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = ((dst * (256 - a) + src[..., 2::-1] * a) >> 8).astype(np.uint8)


class HudRenderer:
    def __init__(self, W, H, players, rallies, hits, shuttle, fps):
        self.W, self.H, self.fps = W, H, fps
        self.s = min(W, H) / 720
        self.players = players
        self.rallies = rallies
        self.hits = sorted(hits, key=lambda h: h.frame)
        self.shuttle = shuttle
        self.hud_w = int(min(W - 28 * self.s, 700 * self.s))
        self.color = {p.pid: COLORS[i % 4] for i, p in enumerate(players)}
        self.tags = {p.pid: self._name_tag(p.name, self.color[p.pid]) for p in players}
        self.hit_frames = np.array([h.frame for h in self.hits]) if self.hits else np.array([], int)

    # ---------- trạng thái tại khung f
    def _rally_state(self, f):
        cur, prev = None, None
        for r in self.rallies:
            if r.start <= f <= r.end:
                cur = r
            elif r.end < f:
                prev = r
        if cur:
            return (f - cur.start) / self.fps, sum(h.frame <= f for h in cur.hits), True
        if prev:
            return (prev.end - prev.start) / self.fps, len(prev.hits), False
        return 0.0, 0, False

    def _last_hit(self, f):
        k = np.searchsorted(self.hit_frames, f, side="right") - 1
        return self.hits[k] if k >= 0 else None

    # ---------- vẽ
    def draw(self, frame, f):
        s = self.s
        self._draw_shuttle(frame, f)
        for p in self.players:
            if p.seen[f]:
                self._draw_player(frame, p, f)
        hit = self._last_hit(f)
        if hit and hit.speed_kmh and (f - hit.frame) < 1.8 * self.fps:
            p = next((q for q in self.players if q.pid == hit.player), None)
            if p is not None and p.seen[f]:
                x1, y1, x2, y2 = p.box[f]
                blend(frame, self._shot_tag(int(round(hit.speed_kmh))), int(x2 + 6 * s), int(y1 + (y2 - y1) * 0.15))
        blend(frame, self._hud(f, hit), int(14 * s), int(14 * s))

    def _draw_shuttle(self, frame, f):
        s = self.s
        a = max(0, f - 10)
        pts = self.shuttle[a:f + 1]
        prev = None
        for i, p in enumerate(pts):
            if np.isnan(p[0]):
                prev = None
                continue
            q = (int(p[0]), int(p[1]))
            if prev is not None:
                t = (i + 1) / len(pts)
                cv2.line(frame, prev, q, (255, int(208 * t + 47), int(63 * t)), max(1, int(4 * s * t)), cv2.LINE_AA)
            prev = q
        cur = self.shuttle[f]
        if not np.isnan(cur[0]):
            c = (int(cur[0]), int(cur[1]))
            r = int(11 * s)
            cv2.circle(frame, c, r, (255, 230, 120), max(1, int(2 * s)), cv2.LINE_AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(frame, (c[0] + dx * r, c[1] + dy * r), (c[0] + dx * (r + int(7 * s)), c[1] + dy * (r + int(7 * s))),
                         (255, 230, 120), max(1, int(2 * s)), cv2.LINE_AA)

    def _draw_player(self, frame, p, f):
        s = self.s
        x1, y1, x2, y2 = p.box[f]
        col = self.color[p.pid][::-1]
        cx = int((x1 + x2) / 2)
        w = x2 - x1
        cv2.ellipse(frame, (cx, int(y2)), (max(int(14 * s), int(w * 0.45)), max(int(5 * s), int(w * 0.14))),
                    0, 0, 360, col, max(1, int(2.5 * s)), cv2.LINE_AA)
        tag = self.tags[p.pid]
        blend(frame, tag, cx - tag.shape[1] // 2, int(y1 - tag.shape[0] - 8 * s))

    def _name_tag(self, name, color):
        s = self.s
        fnt = font(15 * s)
        text = name.upper()
        tw = int(fnt.getlength(text) + 18 * s)
        th = int(15 * s + 12 * s)
        img = Image.new("RGBA", (tw, th + int(7 * s)), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, tw - 1, th - 1], radius=int(5 * s), fill=PANEL, outline=color + (255,), width=max(1, int(1.5 * s)))
        d.polygon([(tw / 2 - 5 * s, th - 1), (tw / 2 + 5 * s, th - 1), (tw / 2, th + 6 * s)], fill=color + (255,))
        d.text((tw / 2, th / 2), text, font=fnt, fill=color + (255,), anchor="mm")
        return np.array(img)

    @lru_cache(maxsize=512)
    def _shot_tag(self, kmh):
        s = self.s
        fnt, lab = font(14 * s), font(9 * s)
        text = f"{kmh} km/h"
        tw = int(fnt.getlength(text) + 52 * s)
        th = int(14 * s + 12 * s)
        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, tw - 1, th - 1], radius=int(5 * s), fill=PANEL, outline=ACCENT + (255,), width=max(1, int(1.5 * s)))
        d.text((8 * s, th / 2), "SHOT", font=lab, fill=LABEL, anchor="lm")
        d.text((44 * s, th / 2), text, font=fnt, fill=WHITE, anchor="lm")
        return np.array(img)

    def _hud(self, f, hit):
        s, hw = self.s, self.hud_w
        gap, top_h = 8 * s, 74 * s
        row_h = 44 * s
        tbl_h = row_h * len(self.players) + 8 * s
        img = Image.new("RGBA", (hw + 2, int(top_h + gap + tbl_h + 2)), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        rally_t, count, active = self._rally_state(f)
        flash = hit is not None and 0 <= f - hit.frame < 0.4 * self.fps
        speed = next((h.speed_kmh for h in reversed(self.hits[:np.searchsorted(self.hit_frames, f, side="right")]) if h.speed_kmh), 0)
        w1, w2 = hw * 0.4, hw * 0.27
        w3 = hw - w1 - w2 - 2 * gap
        self._stat(d, 0, 0, w1, top_h, "RALLY TIME", fmt_clock(rally_t), "", False)
        self._stat(d, w1 + gap, 0, w2, top_h, "RALLY COUNT", str(count), "SHOTS", flash)
        self._stat(d, w1 + w2 + 2 * gap, 0, w3, top_h, "SHOT SPEED", str(int(round(speed))) if speed else "--", "KM/H", False)

        y2 = top_h + gap
        tw = hw * 0.5
        self._panel(d, 0, y2, tw, tbl_h)
        for i, p in enumerate(self.players):
            col = self.color[p.pid] + (255,)
            ry = y2 + 4 * s + i * row_h
            d.rectangle([3 * s, ry + 6 * s, 6 * s, ry + row_h - 6 * s], fill=col)
            d.text((10 * s, ry + 4 * s), p.name.upper(), font=font(17 * s), fill=col)
            stamina = int(round(p.stamina[f]))
            self._labeled(d, "STAMINA", str(stamina), tw - 10 * s, ry + 4 * s, 17 * s)
            n = 20
            filled = round(stamina / 100 * n)
            bw = tw * 0.62
            for k in range(n):
                x = 10 * s + k * bw / n
                d.rectangle([x, ry + 27 * s, x + bw / n - 1.5 * s, ry + 33 * s], fill=col if k < filled else (255, 255, 255, 30))
            self._labeled(d, "MOVE", f"{p.dist[f]:.1f} m", tw - 10 * s, ry + 26 * s, 12 * s)

        self._court_map(d, tw + gap, y2, hw - tw - gap, tbl_h, f)
        return np.array(img)

    def _panel(self, d, x, y, w, h, hl=False):
        d.rounded_rectangle([x, y, x + w, y + h], radius=int(8 * self.s), fill=PANEL,
                            outline=(ACCENT + (255,)) if hl else EDGE, width=max(1, int(1.2 * self.s)))

    def _stat(self, d, x, y, w, h, label, value, unit, hl):
        s = self.s
        self._panel(d, x, y, w, h, hl)
        d.text((x + 12 * s, y + 8 * s), label, font=font(12 * s), fill=LABEL)
        big = font(34 * s)
        d.text((x + 12 * s, y + h - 8 * s), value, font=big, fill=WHITE, anchor="ls")
        if unit:
            d.text((x + 16 * s + big.getlength(value), y + h - 10 * s), unit, font=font(12 * s), fill=LABEL, anchor="ls")

    def _labeled(self, d, label, value, right, y, size):
        f = font(size)
        d.text((right, y), value, font=f, fill=WHITE, anchor="ra")
        d.text((right - f.getlength(value) - 6 * self.s, y + size * 0.35), label, font=font(9 * self.s), fill=LABEL, anchor="ra")

    def _court_map(self, d, x, y, w, h, f):
        s = self.s
        self._panel(d, x, y, w, h)
        d.text((x + 10 * s, y + 8 * s), "COURT MAP", font=font(11 * s), fill=LABEL)
        d.text((x + w - 10 * s, y + 9 * s), "PLAYER POSITIONS", font=font(9 * s), fill=LABEL, anchor="ra")
        ax, ay, aw, ah = x + 10 * s, y + 26 * s, w - 20 * s, h - 46 * s
        k = min(aw / C.L, ah / C.W)
        cw, ch = C.L * k, C.W * k
        ox, oy = ax + (aw - cw) / 2, ay + (ah - ch) / 2

        def mp(cx, cy):
            return ox + (C.L - cy) / C.L * cw, oy + cx / C.W * ch

        d.rectangle([ox, oy, ox + cw, oy + ch], fill=(30, 110, 90, 110))
        for x1, y1, x2, y2 in C.court_lines():
            d.line([mp(x1, y1), mp(x2, y2)], fill=(255, 255, 255, 140), width=max(1, int(s)))
        d.line([mp(0, C.NET), mp(C.W, C.NET)], fill=ACCENT + (255,), width=max(1, int(2 * s)))
        trail = int(1.5 * self.fps)
        for p in self.players:
            col = self.color[p.pid]
            seg = p.pos[max(0, f - trail):f + 1]
            pts = [mp(*q) for q in seg if not np.isnan(q[0])]
            if len(pts) > 1:
                d.line(pts, fill=col + (130,), width=max(1, int(1.5 * s)))
            if not np.isnan(p.pos[f, 0]):
                px, py = mp(*p.pos[f])
                r = 5 * s
                d.ellipse([px - r, py - r, px + r, py + r], fill=col + (255,), outline=WHITE, width=max(1, int(1.2 * s)))
        d.text((x + 10 * s, y + h - 6 * s), "OUR TEAM", font=font(9 * s), fill=LABEL, anchor="ls")
        d.text((x + w - 10 * s, y + h - 6 * s), "OPPONENTS", font=font(9 * s), fill=LABEL, anchor="rs")


SYNC_BACK = 90  # số khung phân tích phía trước a0 dùng để dò vị trí sau khi tua


def thumb(frame):
    """'Dấu vân tay' 64x36 của một khung — để tìm lại đúng khung sau khi tua video."""
    return cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def _open(video, threads=3):
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads])
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(str(video))


def _open_at(video, start_f, step, a0, ref_thumbs):
    """Mở video sao cho lần read() tiếp theo trả về khung phân tích a0.

    Tua bằng OpenCV có thể lệch vài khung, nên tua về trước một đoạn rồi đọc tới, so dấu vân tay với
    các khung đã lưu ở lượt nhận diện để biết chính xác mình đang ở khung nào.
    """
    cap = _open(video)
    if a0 == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        return cap
    lo = a0 - len(ref_thumbs)
    ref = ref_thumbs.astype(np.int16)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(start_f, start_f + (a0 - len(ref_thumbs) // 2) * step))
    for _ in range(len(ref_thumbs) * step * 2):
        ok, f = cap.read()
        if not ok:
            break
        d = np.abs(ref - thumb(f)).mean((1, 2))
        k = int(d.argmin())
        if d[k] < 2.5:
            for _ in range((a0 - (lo + k)) * step - 1):
                cap.grab()
            return cap
    # Không dò được (video lạ): đọc tuần tự từ đầu, chậm nhưng chắc chắn đúng.
    cap.release()
    cap = _open(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    for _ in range(a0 * step):
        cap.grab()
    return cap


def _render_chunk(job):
    """Tiến trình con: nhảy tới đoạn của mình, vẽ HUD, ghi file tạm."""
    (video, start_f, step, a0, a1, out_path, state, ref_thumbs) = job
    cv2.setNumThreads(2)
    hud = HudRenderer(*state)
    W, H, fps = hud.W, hud.H, hud.fps
    cap = _open_at(video, start_f, step, a0, ref_thumbs)
    writer = open_writer(out_path, W, H, fps, threads=3)
    for a in range(a0, a1):
        ok, frame = cap.read()
        if not ok:
            break
        for _ in range(step - 1):
            cap.grab()
        hud.draw(frame, a)
        writer.stdin.write(frame.tobytes())
    writer.stdin.close()
    writer.wait()
    cap.release()
    return a1 - a0


def render_video(out_path, video, start_f, step, n, state, thumbs, start_s, dur_s, workers=4, on_progress=None):
    """Chia video thành nhiều đoạn, vẽ song song trên nhiều tiến trình rồi ghép lại kèm âm thanh gốc."""
    import tempfile
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="cv_render_"))
    bounds = np.linspace(0, n, workers + 1).astype(int)
    jobs = [(video, start_f, step, int(bounds[i]), int(bounds[i + 1]), tmp / f"part{i}.mp4", state,
             thumbs[max(0, bounds[i] - SYNC_BACK):bounds[i]])
            for i in range(workers) if bounds[i + 1] > bounds[i]]
    with ProcessPoolExecutor(len(jobs)) as ex:
        futs = [ex.submit(_render_chunk, j) for j in jobs]
        for fut in as_completed(futs):
            done = fut.result()
            if on_progress:
                on_progress(done)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{j[5].as_posix()}'\n" for j in jobs), encoding="utf-8")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
           "-ss", f"{start_s:.3f}"] + (["-t", f"{dur_s:.3f}"] if dur_s else []) + \
          ["-i", str(video), "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
           "-shortest", "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


def open_writer(path, W, H, fps, audio_src=None, start_s=0.0, dur_s=None, threads=0):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", f"{fps:.4f}", "-i", "-"]
    if audio_src:
        cmd += ["-ss", f"{start_s:.3f}"] + (["-t", f"{dur_s:.3f}"] if dur_s else []) + ["-i", str(audio_src),
                                                                                        "-map", "0:v", "-map", "1:a?"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p"]
    if threads:
        cmd += ["-threads", str(threads)]
    if audio_src:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd += ["-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
