"""Từ quỹ đạo cầu + vị trí người chơi → cú đánh, rally, tốc độ cầu, quãng đường, thể lực."""
from dataclasses import dataclass, field

import numpy as np

import court as C
from players import L_WRIST, R_WRIST

SHUTTLE_K = 9.81 / (6.7 * 6.7)  # lực cản cầu lông: vận tốc rơi tới hạn ~6.7 m/s


# ---------------------------------------------------------------- quả cầu
def clean_shuttle(raw, n_frames, frame_h):
    """raw: {frame: (x, y) | None} → mảng (N, 2) đã lọc nhiễu, NaN = không thấy."""
    p = np.full((n_frames, 2), np.nan)
    for f, v in raw.items():
        if v is not None and f < n_frames:
            p[f] = v
    valid = ~np.isnan(p[:, 0])

    # 1) Điểm đứng yên lâu (chữ/logo trên video, cầu nằm dưới sàn) → bỏ.
    idx = np.flatnonzero(valid)
    run_start = 0
    for k in range(1, len(idx) + 1):
        moving = k < len(idx) and np.hypot(*(p[idx[k]] - p[idx[run_start]])) > 4
        if k == len(idx) or moving:
            if idx[k - 1] - idx[run_start] >= 12:
                p[idx[run_start]:idx[k - 1] + 1] = np.nan
            run_start = k
    valid = ~np.isnan(p[:, 0])

    # 2) Điểm nhảy xa bất thường so với cả trước lẫn sau → nhận nhầm.
    thr = 0.09 * frame_h
    idx = np.flatnonzero(valid)
    for a, b, c in zip(idx[:-2], idx[1:-1], idx[2:]):
        if b - a <= 3 and c - b <= 3:
            if np.hypot(*(p[b] - p[a])) > thr * (b - a) and np.hypot(*(p[c] - p[b])) > thr * (c - b):
                p[b] = np.nan
    # Điểm lẻ loi (không có hàng xóm trong 4 khung) cũng bỏ.
    valid = ~np.isnan(p[:, 0])
    idx = np.flatnonzero(valid)
    for k, i in enumerate(idx):
        near = (k > 0 and i - idx[k - 1] <= 4) or (k + 1 < len(idx) and idx[k + 1] - i <= 4)
        if not near:
            p[i] = np.nan

    # 3) Lấp khoảng trống ngắn (≤ 4 khung) bằng nội suy.
    valid = ~np.isnan(p[:, 0])
    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 5:
            t = np.linspace(0, 1, b - a + 1)[1:-1, None]
            p[a + 1:b] = p[a] * (1 - t) + p[b] * t
    return p


# ---------------------------------------------------------------- người chơi
@dataclass
class PlayerSeries:
    pid: int
    name: str
    side: int
    pos: np.ndarray       # (N, 2) mét, NaN khi mất dấu
    box: np.ndarray       # (N, 4)
    wrist: np.ndarray     # (N, 2, 2) hai cổ tay (ảnh)
    seen: np.ndarray      # (N,) bool — thấy trực tiếp hoặc nội suy ngắn
    dist: np.ndarray = None      # quãng đường cộng dồn (m)
    speed: np.ndarray = None     # m/s
    stamina: np.ndarray = None
    hits: list = field(default_factory=list)


def build_player_series(track, n_frames, fps):
    keys = np.array(sorted(track.frames))
    pos = np.full((n_frames, 2), np.nan)
    box = np.full((n_frames, 4), np.nan)
    wrist = np.full((n_frames, 2, 2), np.nan)
    seen = np.zeros(n_frames, bool)
    if len(keys):
        P = np.array([track.frames[k].pos for k in keys])
        B = np.array([track.frames[k].box for k in keys])
        Wr = np.array([track.frames[k].kps[[L_WRIST, R_WRIST], :2] for k in keys])
        # Làm mượt vị trí (trung bình trượt 5 mẫu) để quãng đường không bị cộng nhiễu.
        if len(P) >= 5:
            ker = np.ones(5) / 5
            P = np.stack([np.convolve(np.pad(P[:, d], 2, mode="edge"), ker, "valid") for d in range(2)], 1)
        max_gap = int(fps * 1.0)
        for a in range(len(keys)):
            f0 = keys[a]
            pos[f0], box[f0], wrist[f0], seen[f0] = P[a], B[a], Wr[a], True
            if a + 1 < len(keys) and keys[a + 1] - f0 <= max_gap:
                f1 = keys[a + 1]
                t = np.linspace(0, 1, f1 - f0 + 1)[1:-1]
                pos[f0 + 1:f1] = P[a] + (P[a + 1] - P[a]) * t[:, None]
                box[f0 + 1:f1] = B[a] + (B[a + 1] - B[a]) * t[:, None]
                wrist[f0 + 1:f1] = Wr[a] + (Wr[a + 1] - Wr[a]) * t[:, None, None]
                seen[f0 + 1:f1] = True
    s = PlayerSeries(track.pid, track.name, track.side, pos, box, wrist, seen)
    _motion_stats(s, fps)
    return s


def _motion_stats(s, fps):
    n = len(s.pos)
    step = np.zeros(n)
    d = np.diff(s.pos, axis=0)
    ln = np.hypot(d[:, 0], d[:, 1])
    ok = ~np.isnan(ln) & (ln * fps < 9.0)          # bỏ bước nhảy > 9 m/s (nhận nhầm)
    step[1:][ok] = ln[ok]
    s.dist = np.cumsum(step)
    # Tốc độ: trung bình trượt 0.3 s
    win = max(1, int(fps * 0.3))
    s.speed = np.convolve(step, np.ones(win) / win, "same") * fps
    # Thể lực: tốn khi di chuyển/bứt tốc, hồi khi đứng yên.
    st = np.empty(n)
    v = 100.0
    for i in range(n):
        sp = s.speed[i]
        v -= step[i] * 0.22
        if sp > 3:
            v -= (sp - 3) * 0.5 / fps
        elif sp < 0.7:
            v += 0.9 / fps
        v = min(100.0, max(5.0, v))
        st[i] = v
    s.stamina = st


# ---------------------------------------------------------------- âm thanh
def audio_onsets(pcm, sr, sensitivity=6):
    """Thời điểm (giây) các tiếng 'bốp' — cùng thuật toán với app điện thoại, viết dạng vector."""
    from scipy.signal import lfilter, butter
    if pcm is None or len(pcm) < sr:
        return np.array([])
    b, a = butter(2, 2200 / (sr / 2), "highpass")
    e = lfilter(b, a, pcm) ** 2
    af = 1 - np.exp(-1 / (0.003 * sr))
    aslow = 1 - np.exp(-1 / (0.6 * sr))
    fast = lfilter([af], [1, af - 1], e)
    slow = lfilter([aslow], [1, aslow - 1], np.minimum(e, np.percentile(e, 99))) + 1e-9
    ratio = 16 - sensitivity * 1.3
    cand = np.flatnonzero((fast > ratio * slow) & (fast > 2e-5))
    cand = cand[cand > 0.5 * sr]
    out, last = [], -1e9
    for i in cand:
        if i - last > 0.18 * sr:
            out.append(i / sr)
            last = i
    return np.array(out)


# ---------------------------------------------------------------- cú đánh & rally
@dataclass
class Hit:
    frame: int
    player: int           # pid, -1 nếu không rõ
    side: int
    speed_kmh: float = 0.0


@dataclass
class Rally:
    start: int
    end: int
    hits: list
    land_court: tuple = None   # vị trí cầu rơi (m) nếu thấy
    land_in: bool = None


def detect_hits(shuttle, players, fps, onsets=None):
    n = len(shuttle)
    valid = ~np.isnan(shuttle[:, 0])
    cands = []
    for t in range(4, n - 4):
        if not valid[t]:
            continue
        before = shuttle[t - 3:t]
        after = shuttle[t + 1:t + 4]
        if np.isnan(before).any() or np.isnan(after).any():
            continue
        v1 = shuttle[t] - before.mean(0)
        v2 = after.mean(0) - shuttle[t]
        s1, s2 = np.hypot(*v1), np.hypot(*v2)
        if s2 < 3 or s1 < 1:
            continue
        cos = np.dot(v1, v2) / (s1 * s2)
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        accel = s2 / max(s1, 1e-3)
        score = angle / 45 + max(0, accel - 1.5)
        if angle < 50 and accel < 2.2:
            continue
        who, d = _nearest_player(shuttle[t], players, t)
        if who is None:
            continue
        cands.append((t, score, who))

    # Một cú đánh có thể sinh nhiều ứng viên liền nhau → giữ cái mạnh nhất trong 8 khung.
    cands.sort(key=lambda c: -c[1])
    taken = []
    for t, sc, who in cands:
        if all(abs(t - u) > 8 for u, _, _ in taken):
            taken.append((t, sc, who))
    taken.sort()

    hits = []
    for t, sc, who in taken:
        if onsets is not None and len(onsets):
            k = np.abs(onsets * fps - t).argmin()
            if abs(onsets[k] * fps - t) <= 4:
                t = int(round(onsets[k] * fps))  # khớp đúng khung có tiếng vợt
        p = players[who]
        # Hai bên luân phiên: cùng phía, quá gần nhau → trùng một cú.
        if hits and hits[-1].side == p.side and t - hits[-1].frame < 0.45 * fps:
            continue
        hits.append(Hit(t, p.pid, p.side))
    return hits


def _nearest_player(pt, players, t):
    best, bd = None, 1e9
    for i, p in enumerate(players):
        if not p.seen[t]:
            continue
        x1, y1, x2, y2 = p.box[t]
        w, h = x2 - x1, y2 - y1
        if not (x1 - 0.8 * w <= pt[0] <= x2 + 0.8 * w and y1 - 0.7 * h <= pt[1] <= y2 - 0.25 * h):
            continue
        wr = p.wrist[t]
        d = min(np.hypot(*(pt - wr[0])), np.hypot(*(pt - wr[1])), np.hypot(pt[0] - (x1 + x2) / 2, pt[1] - y1))
        if d < bd:
            best, bd = i, d
    return best, bd


def estimate_speeds(hits, shuttle, players, court, fps):
    by_pid = {p.pid: p for p in players}
    for k, h in enumerate(hits):
        p = by_pid.get(h.player)
        if p is None or np.isnan(p.pos[h.frame]).any():
            continue
        v_img = 0.0
        seg = shuttle[h.frame + 1:h.frame + 6]
        if not np.isnan(seg).any():
            steps = np.diff(seg, axis=0)
            lens = np.hypot(steps[:, 0], steps[:, 1])
            dirs = steps / np.maximum(lens[:, None], 1e-6)
            # Chỉ tin khi 5 khung sau cú đánh bay thẳng một hướng (không phải nhận nhầm nhảy lung tung).
            if lens.min() > 2 and (dirs[1:] * dirs[:-1]).sum(1).min() > 0.8:
                ppm = court.px_per_meter(p.pos[h.frame])
                if ppm > 1:
                    v_img = float(np.median(lens)) * fps / ppm  # m/s, chỉ phần song song mặt ảnh
        v_flight = 0.0
        if k + 1 < len(hits):
            nxt = hits[k + 1]
            q = by_pid.get(nxt.player)
            dt = (nxt.frame - h.frame) / fps
            if q is not None and nxt.side != h.side and 0.2 < dt < 3 and not np.isnan(q.pos[nxt.frame]).any():
                x = np.hypot(*(q.pos[nxt.frame] - p.pos[h.frame]))
                if x > 1:
                    v_flight = (np.exp(SHUTTLE_K * x) - 1) / (SHUTTLE_K * dt)
        # Ưu tiên ước tính theo thời gian bay (bền hơn); số đo trên ảnh chỉ dùng khi lệch không quá xa.
        if v_flight and v_img:
            v = max(v_flight, min(v_img * 1.15, v_flight * 1.6))
        else:
            v = v_flight or v_img * 1.15
        if v > 0:
            h.speed_kmh = float(np.clip(v * 3.6, 15, 350))


def build_rallies(hits, shuttle, court, fps, gap_s=3.0):
    rallies, cur = [], []
    for h in hits:
        if cur and (h.frame - cur[-1].frame) / fps > gap_s:
            rallies.append(cur)
            cur = []
        cur.append(h)
    if cur:
        rallies.append(cur)
    out = []
    n = len(shuttle)
    for hs in rallies:
        last = hs[-1].frame
        end, land = min(n - 1, last + int(2.5 * fps)), None
        # Cầu rơi: điểm cuối cùng còn thấy trước khi mất hẳn (≥ 10 khung).
        miss = 0
        for f in range(last + 1, min(n, last + int(3 * fps))):
            if np.isnan(shuttle[f, 0]):
                miss += 1
                if miss >= 10:
                    end = f - miss
                    break
            else:
                miss = 0
                land = shuttle[f]
        r = Rally(hs[0].frame, max(end, last + int(0.3 * fps)), hs)
        if land is not None:
            lx, ly = court.to_court([land])[0]
            r.land_court = (float(lx), float(ly))
            r.land_in = bool(-0.1 <= lx <= C.W + 0.1 and -0.1 <= ly <= C.L + 0.1)
        out.append(r)
    return out
