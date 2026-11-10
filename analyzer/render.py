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
    # Phép tính của OpenCV (SIMD, đa luồng) nhanh hơn numpy ~10 lần: out = dst·(1-a) + src·a
    a = cv2.cvtColor(np.ascontiguousarray(src[..., 3]), cv2.COLOR_GRAY2BGR)
    bgr = cv2.cvtColor(np.ascontiguousarray(src[..., :3]), cv2.COLOR_RGB2BGR)
    dst = frame[y0:y1, x0:x1]
    keep = cv2.multiply(dst, cv2.bitwise_not(a), scale=1 / 255)
    frame[y0:y1, x0:x1] = cv2.add(keep, cv2.multiply(bgr, a, scale=1 / 255))


SHIFT = 4  # toạ độ lẻ 1/16 pixel khi vẽ → nét không giật theo từng pixel
RING_M = 0.42  # bán kính vòng khoá người trên mặt sân (m)


def _fx(pts):
    return np.round(np.asarray(pts, np.float64) * (1 << SHIFT)).astype(np.int32)


def _smooth(arr, ok, win):
    """Làm mượt theo thời gian (Gauss, bỏ qua khung mất dấu)."""
    a = arr.reshape(len(arr), -1).astype(np.float64)
    ok = ok & ~np.isnan(a).any(1)
    x = np.arange(-win, win + 1)
    ker = np.exp(-0.5 * (x / max(1.0, win / 2)) ** 2)
    den = np.convolve(ok.astype(np.float64), ker, "same")
    out = np.full_like(a, np.nan)
    for d in range(a.shape[1]):
        num = np.convolve(np.where(ok, a[:, d], 0.0), ker, "same")
        out[:, d] = np.where(ok, num / np.maximum(den, 1e-9), np.nan)
    return out.reshape(arr.shape)


def _project(court, pts, limit):
    """Chiếu điểm mặt sân (m) lên ảnh; bỏ điểm sau lưng camera hoặc văng quá xa khung hình."""
    p = np.c_[np.asarray(pts, np.float64), np.ones(len(pts))] @ court.Hinv.T
    ok = p[:, 2] > 1e-6
    xy = np.full((len(pts), 2), np.nan)
    xy[ok] = p[ok, :2] / p[ok, 2:3]
    ok &= (np.abs(np.nan_to_num(xy, nan=1e9)) < limit).all(1)
    return xy, ok


def _runs(xy, ok):
    """Tách dãy điểm thành các đoạn liền nhau còn nhìn thấy."""
    out, cur = [], []
    for q, k in zip(xy, ok):
        if k:
            cur.append(q)
        else:
            if len(cur) > 1:
                out.append(np.array(cur))
            cur = []
    if len(cur) > 1:
        out.append(np.array(cur))
    return out


def _paint(frame, polys, color, alpha, thick=1, closed=False, fill=False):
    """Vẽ nét/đa giác bán trong suốt, chỉ xử lý vùng ảnh bao quanh hình."""
    if alpha <= 0.02 or not polys:
        return
    allp = np.concatenate(polys)
    H, W = frame.shape[:2]
    pad = thick + 2
    x0, y0 = np.floor(allp.min(0)).astype(int) - pad
    x1, y1 = np.ceil(allp.max(0)).astype(int) + pad
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x0 >= x1 or y0 >= y1:
        return
    crop = frame[y0:y1, x0:x1]
    ov = crop.copy()
    pts = [_fx(q - (x0, y0)) for q in polys]
    if fill:
        cv2.fillPoly(ov, pts, color, cv2.LINE_AA, SHIFT)
    else:
        cv2.polylines(ov, pts, closed, color, thick, cv2.LINE_AA, SHIFT)
    cv2.addWeighted(ov, alpha, crop, 1 - alpha, 0, dst=crop)


class HudRenderer:
    def __init__(self, W, H, players, rallies, hits, shuttle, fps, courts=None):
        self.W, self.H, self.fps = W, H, fps
        self.s = min(W, H) / 720
        self.players = players
        self.rallies = rallies
        self.hits = sorted(hits, key=lambda h: h.frame)
        self.shuttle = shuttle
        self.courts = courts
        self.hud_w = int(min(W - 28 * self.s, 500 * self.s))
        self.color = {p.pid: COLORS[i % 4] for i, p in enumerate(players)}
        self.tags = {p.pid: self._name_tag(p.name, self.color[p.pid]) for p in players}
        self.hit_frames = np.array([h.frame for h in self.hits]) if self.hits else np.array([], int)
        # Vị trí/khung người đã làm mượt + độ đậm (mờ dần khi vừa thấy / sắp mất dấu) → vòng không giật, không chớp.
        win = max(2, int(round(fps * 0.16)))
        fade = np.ones(2 * max(2, int(round(fps * 0.2))) + 1)
        self.spos, self.sbox, self.vis = {}, {}, {}
        for p in players:
            self.spos[p.pid] = _smooth(p.pos, p.seen, win)
            self.sbox[p.pid] = _smooth(p.box, p.seen, win)
            v = np.convolve(p.seen.astype(np.float64), fade / fade.sum(), "same")
            self.vis[p.pid] = np.clip((v - 0.5) * 2.2, 0, 1) * p.seen
        ang = np.radians(np.arange(0, 361, 6))
        self.circle = np.c_[np.cos(ang), np.sin(ang)] * RING_M
        self.arcs = [np.c_[np.cos(t), np.sin(t)] * RING_M
                     for t in (np.radians(np.arange(c - 30, c + 31, 5)) for c in (45, 135, 225, 315))]
        self._court_cache = {}

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
        court = self.courts.at(f) if self.courts is not None else None
        if court is not None:
            self._draw_court(frame, f)
        for p in self.players:
            if self.vis[p.pid][f] > 0.02:
                self._draw_player(frame, p, f, court)
        self._draw_shuttle(frame, f)
        hit = self._last_hit(f)
        if hit and hit.speed_kmh and (f - hit.frame) < 1.8 * self.fps:
            p = next((q for q in self.players if q.pid == hit.player), None)
            if p is not None and self.vis[p.pid][f] > 0.5 and not np.isnan(self.sbox[p.pid][f, 0]):
                x1, y1, x2, y2 = self.sbox[p.pid][f]
                tag = self.tags[p.pid]
                blend(frame, self._shot_tag(int(round(hit.speed_kmh))), int(round((x1 + x2) / 2 + tag.shape[1] / 2 + 4 * s)),
                      int(round(y1 - tag.shape[0] - 6 * s)))
        blend(frame, self._hud(f, hit), int(14 * s), int(14 * s))

    def _draw_court(self, frame, f):
        """Khung sân tối giản: nền phủ rất nhạt, viền mảnh, 4 góc khoá và dấu chân lưới."""
        k = self.courts.index(f) if len(self.courts.times) > 1 else 0
        hit = self._court_cache.get(k)
        if hit is None:
            if len(self._court_cache) > 24:
                self._court_cache.clear()
            hit = self._court_cache[k] = self._court_layer(self.courts.at(f))
        if hit[0] is not None:
            blend(frame, *hit)

    def _court_layer(self, court):
        s, W, H = self.s, self.W, self.H
        lim = 4 * max(W, H)
        acc = ACCENT

        def line(a, b, n=48):
            t = np.linspace(0, 1, n)[:, None]
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            return _runs(*_project(court, a + (b - a) * t, lim))

        cs = [(0, 0), (C.W, 0), (C.W, C.L), (0, C.L)]
        layer = np.zeros((H, W, 4), np.uint8)
        t = np.linspace(0, 1, 48, endpoint=False)[:, None]
        ring = np.concatenate([np.asarray(cs[i], np.float64) + (np.asarray(cs[(i + 1) % 4]) - np.asarray(cs[i])) * t
                               for i in range(4)])
        xy, ok = _project(court, ring, lim)
        if ok.sum() >= 3:
            cv2.fillPoly(layer, [_fx(xy[ok])], acc + (20,), cv2.LINE_AA, SHIFT)
        for i in range(4):
            for run in line(cs[i], cs[(i + 1) % 4]):
                cv2.polylines(layer, [_fx(run)], False, acc + (90,), max(1, int(round(1.0 * s))), cv2.LINE_AA, SHIFT)
        leg = 0.9
        for cx, cy in cs:   # góc khoá chữ L
            dx = leg if cx == 0 else -leg
            dy = leg if cy == 0 else -leg
            # Máy đặt thấp: góc ở xa bị ép dẹt thành một vệt → bỏ, tránh rối.
            e, eok = _project(court, [(cx, cy), (cx + dx, cy), (cx, cy + dy)], lim)
            if not eok.all() or np.hypot(*(e[2] - e[0])) < 0.45 * np.hypot(*(e[1] - e[0])):
                continue
            for a, b in (((cx, cy), (cx + dx, cy)), ((cx, cy), (cx, cy + dy))):
                for run in line(a, b, 12):
                    cv2.polylines(layer, [_fx(run)], False, acc + (230,), max(1, int(round(1.6 * s))),
                                  cv2.LINE_AA, SHIFT)
        for a, b in (((0, C.NET), (0.55, C.NET)), ((C.W, C.NET), (C.W - 0.55, C.NET)),
                     ((0, C.NET - 0.4), (0, C.NET + 0.4)), ((C.W, C.NET - 0.4), (C.W, C.NET + 0.4))):
            for run in line(a, b, 8):       # dấu chân lưới hai bên
                cv2.polylines(layer, [_fx(run)], False, acc + (230,), max(1, int(round(1.6 * s))), cv2.LINE_AA, SHIFT)
        ys, xs = np.nonzero(layer[..., 3])
        if not len(xs):
            return None, 0, 0
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        return np.ascontiguousarray(layer[y0:y1, x0:x1]), int(x0), int(y0)

    def _draw_shuttle(self, frame, f):
        """Cầu: chấm sáng nhỏ có quầng mờ + vệt đuôi thon dần."""
        s = self.s
        n_tail = max(6, int(self.fps * 0.45))
        a = max(0, f - n_tail)
        pts = self.shuttle[a:f + 1]
        ok = ~np.isnan(pts[:, 0])
        if not ok.any() or (len(pts) - 1 - np.nonzero(ok)[0][-1]) > 2:
            return
        pad = int(14 * s) + 2
        x0, y0 = np.floor(pts[ok].min(0)).astype(int) - pad
        x1, y1 = np.ceil(pts[ok].max(0)).astype(int) + pad
        w, h = int(x1 - x0), int(y1 - y0)
        if w <= 0 or h <= 0 or w * h > 4_000_000:
            return
        m = np.zeros((h, w), np.uint8)
        off = np.array([x0, y0])
        for i in range(1, len(pts)):
            if ok[i] and ok[i - 1]:
                t = i / (len(pts) - 1)
                cv2.line(m, tuple(int(v) for v in _fx(pts[i - 1] - off)), tuple(int(v) for v in _fx(pts[i] - off)),
                         int(40 + 170 * t ** 1.6), max(1, int(round(3.2 * s * t))), cv2.LINE_AA, SHIFT)
        rgba = np.empty((h, w, 4), np.uint8)
        rgba[..., :3] = (255, 226, 130)
        if ok[-1]:
            c = tuple(int(v) for v in _fx(pts[-1] - off))
            g = np.zeros((h, w), np.uint8)
            cv2.circle(g, c, int(round(7.5 * s * 16)), 120, -1, cv2.LINE_AA, SHIFT)
            kk = 2 * int(3 * s) + 3
            m = np.maximum(m, cv2.GaussianBlur(g, (kk, kk), 0))
            core = np.zeros((h, w), np.uint8)
            cv2.circle(core, c, int(round(3.0 * s * 16)), 255, -1, cv2.LINE_AA, SHIFT)
            m = np.maximum(m, core)
            rgba[..., :3][core > 128] = (255, 255, 255)
        rgba[..., 3] = m
        blend(frame, rgba, int(x0), int(y0))

    def _draw_player(self, frame, p, f, court):
        s = self.s
        vis = float(self.vis[p.pid][f])
        col = self.color[p.pid][::-1]
        x1, y1, x2, y2 = self.sbox[p.pid][f]
        cx = (x1 + x2) / 2
        drawn = False
        if court is not None and not np.isnan(self.spos[p.pid][f, 0]):
            # Vòng tròn bán kính cố định (m) nằm trên mặt sân, chiếu theo phối cảnh → luôn "dính" sàn.
            lim = 4 * max(self.W, self.H)
            xy, ok = _project(court, self.spos[p.pid][f] + self.circle, lim)
            if ok.all():
                _paint(frame, [xy], col, 0.16 * vis, fill=True)
                _paint(frame, [xy], col, 0.45 * vis, max(1, int(round(1.0 * s))), closed=True)
                arcs = [_project(court, self.spos[p.pid][f] + a, lim)[0] for a in self.arcs]
                _paint(frame, arcs, col, 0.95 * vis, max(2, int(round(2.4 * s))))
                drawn = True
        if not drawn and not np.isnan(x1):
            w = x2 - x1
            el = cv2.ellipse2Poly((int(cx), int(y2)), (max(int(10 * s), int(w * 0.32)), max(int(4 * s), int(w * 0.1))),
                                  0, 0, 360, 8)
            _paint(frame, [el.astype(np.float64)], col, 0.9 * vis, max(1, int(round(2 * s))), closed=True)
        if vis > 0.5 and not np.isnan(x1):
            tag = self.tags[p.pid]
            blend(frame, tag, int(round(cx - tag.shape[1] / 2)), int(round(y1 - tag.shape[0] - 6 * s)))

    def _name_tag(self, name, color):
        s = self.s
        fnt = font(12 * s)
        text = name.upper()
        bar = 4 * s
        tw = int(fnt.getlength(text) + 16 * s + bar)
        th = int(12 * s + 9 * s)
        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, tw - 1, th - 1], radius=int(th / 2), fill=(8, 18, 28, 170))
        d.rounded_rectangle([5 * s, th * 0.28, 5 * s + bar, th * 0.72], radius=int(bar / 2), fill=color + (255,))
        d.text((5 * s + bar + 5 * s, th / 2), text, font=fnt, fill=WHITE, anchor="lm")
        return np.array(img)

    @lru_cache(maxsize=512)
    def _shot_tag(self, kmh):
        s = self.s
        fnt, lab = font(12 * s), font(9 * s, bold=False)
        text = str(kmh)
        tw = int(fnt.getlength(text) + lab.getlength("km/h") + 19 * s)
        th = int(12 * s + 9 * s)
        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, tw - 1, th - 1], radius=int(th / 2), fill=ACCENT + (235,))
        d.text((8 * s, th / 2), text, font=fnt, fill=(6, 20, 32, 255), anchor="lm")
        d.text((11 * s + fnt.getlength(text), th / 2 + s), "km/h", font=lab, fill=(6, 20, 32, 230), anchor="lm")
        return np.array(img)

    def _hud(self, f, hit):
        """Bảng gọn: một thanh thông số rally, bên dưới là danh sách người chơi và sơ đồ sân nhỏ."""
        s, hw = self.s, self.hud_w
        gap, top_h, row_h = 6 * s, 34 * s, 21 * s
        tbl_h = row_h * len(self.players) + 10 * s
        img = Image.new("RGBA", (hw + 2, int(top_h + gap + tbl_h + 2)), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        rally_t, count, active = self._rally_state(f)
        flash = hit is not None and 0 <= f - hit.frame < 0.4 * self.fps
        speed = next((h.speed_kmh for h in reversed(self.hits[:np.searchsorted(self.hit_frames, f, side="right")]) if h.speed_kmh), 0)
        self._panel(d, 0, 0, hw, top_h)
        mid = top_h / 2
        d.ellipse([11 * s, mid - 3.5 * s, 18 * s, mid + 3.5 * s], fill=(110, 231, 135, 255) if active else (255, 255, 255, 70))
        cells = [("RALLY", fmt_clock(rally_t), "", 0.40), ("SHOTS", str(count), "", 0.25),
                 ("SPEED", str(int(round(speed))) if speed else "--", "km/h", 0.35)]
        x = 26 * s
        span = hw - x
        lab, big = font(9 * s), font(19 * s)
        for i, (label, value, unit, frac) in enumerate(cells):
            if i:
                d.line([x - 10 * s, 9 * s, x - 10 * s, top_h - 9 * s], fill=EDGE, width=max(1, int(s)))
            d.text((x, mid + s), label, font=lab, fill=LABEL, anchor="lm")
            vx = x + lab.getlength(label) + 7 * s
            d.text((vx, mid), value, font=big, fill=(ACCENT + (255,)) if (flash and i == 1) else WHITE, anchor="lm")
            if unit:
                d.text((vx + big.getlength(value) + 4 * s, mid + 3 * s), unit, font=font(9 * s, bold=False), fill=LABEL, anchor="lm")
            x += span * frac

        y2 = top_h + gap
        tw = hw * 0.56
        self._panel(d, 0, y2, tw, tbl_h)
        name_f, val_f, small = font(12 * s), font(11 * s), font(9 * s, bold=False)
        bx0, bx1 = tw * 0.40, tw * 0.66
        for i, p in enumerate(self.players):
            col = self.color[p.pid] + (255,)
            cy = y2 + 5 * s + i * row_h + row_h / 2
            d.rounded_rectangle([8 * s, cy - 6 * s, 11 * s, cy + 6 * s], radius=int(1.5 * s), fill=col)
            d.text((17 * s, cy), p.name.upper(), font=name_f, fill=WHITE, anchor="lm")
            stamina = int(round(p.stamina[f]))
            d.rounded_rectangle([bx0, cy - 1.5 * s, bx1, cy + 1.5 * s], radius=int(1.5 * s), fill=(255, 255, 255, 36))
            d.rounded_rectangle([bx0, cy - 1.5 * s, bx0 + (bx1 - bx0) * max(0.02, stamina / 100), cy + 1.5 * s],
                                radius=int(1.5 * s), fill=col)
            d.text((bx1 + 5 * s, cy), str(stamina), font=small, fill=LABEL, anchor="lm")
            d.text((tw - 9 * s, cy), f"{p.dist[f]:.0f} m", font=val_f, fill=WHITE, anchor="rm")

        self._court_map(d, tw + gap, y2, hw - tw - gap, tbl_h, f)
        return np.array(img)

    def _panel(self, d, x, y, w, h, hl=False):
        d.rounded_rectangle([x, y, x + w, y + h], radius=int(8 * self.s), fill=PANEL,
                            outline=(ACCENT + (255,)) if hl else EDGE, width=max(1, int(1.0 * self.s)))

    def _court_map(self, d, x, y, w, h, f):
        s = self.s
        self._panel(d, x, y, w, h)
        ax, ay, aw, ah = x + 8 * s, y + 8 * s, w - 16 * s, h - 16 * s
        k = min(aw / C.L, ah / C.W)
        cw, ch = C.L * k, C.W * k
        ox, oy = ax + (aw - cw) / 2, ay + (ah - ch) / 2

        def mp(cx, cy):
            return ox + (C.L - cy) / C.L * cw, oy + cx / C.W * ch

        d.rectangle([ox, oy, ox + cw, oy + ch], fill=(30, 110, 90, 90))
        for x1, y1, x2, y2 in C.court_lines():
            d.line([mp(x1, y1), mp(x2, y2)], fill=(255, 255, 255, 90), width=max(1, int(s)))
        d.line([mp(0, C.NET), mp(C.W, C.NET)], fill=ACCENT + (255,), width=max(1, int(1.5 * s)))
        trail = int(1.5 * self.fps)
        for p in self.players:
            col = self.color[p.pid]
            seg = self.spos[p.pid][max(0, f - trail):f + 1]
            pts = [mp(*q) for q in seg if not np.isnan(q[0])]
            if len(pts) > 1:
                d.line(pts, fill=col + (120,), width=max(1, int(1.2 * s)))
            if not np.isnan(seg[-1][0]):
                px, py = mp(*seg[-1])
                r = 3.6 * s
                d.ellipse([px - r, py - r, px + r, py + r], fill=col + (255,), outline=WHITE, width=max(1, int(s)))


SYNC_BACK = 90  # số khung phân tích phía trước a0 dùng để dò vị trí sau khi tua


def thumb(frame):
    """'Dấu vân tay' 64x36 của một khung — để tìm lại đúng khung sau khi tua video."""
    return cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def open_video(video):
    """Mở video, ưu tiên giải mã bằng phần cứng (Intel Quick Sync/D3D11) để dành CPU cho AI."""
    try:
        cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        if cap.isOpened() and cap.read()[0]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap
        cap.release()
    except cv2.error:
        pass
    return cv2.VideoCapture(str(video))


_open = open_video


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
        writer.stdin.write(to_yuv(frame).tobytes())
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


@lru_cache(maxsize=1)
def hw_encoder():
    """Dùng bộ nén video phần cứng Intel Quick Sync nếu có (gần như không tốn CPU), không thì x264."""
    test = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=640x360:d=0.2",
            "-c:v", "h264_qsv", "-f", "null", "-"]
    try:
        ok = subprocess.run(test, capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "23"] if ok else \
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]


def to_yuv(frame):
    """BGR → YUV420 (I420): nhỏ bằng một nửa, đúng định dạng bộ nén cần."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)


def open_writer(path, W, H, fps, audio_src=None, start_s=0.0, dur_s=None, threads=0):
    """Ghi video từ các khung YUV420 (xem to_yuv) đẩy vào stdin."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}",
           "-r", f"{fps:.4f}", "-i", "-"]
    if audio_src:
        cmd += ["-ss", f"{start_s:.3f}"] + (["-t", f"{dur_s:.3f}"] if dur_s else []) + ["-i", str(audio_src),
                                                                                        "-map", "0:v", "-map", "1:a?"]
    enc = hw_encoder()
    cmd += enc + (["-pix_fmt", "nv12"] if "h264_qsv" in enc else ["-pix_fmt", "yuv420p"])
    if threads and "libx264" in enc:
        cmd += ["-threads", str(threads)]
    if audio_src:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd += ["-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
