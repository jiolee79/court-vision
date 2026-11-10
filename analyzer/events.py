"""Từ quỹ đạo cầu + vị trí người chơi → cú đánh, rally, tốc độ cầu, quãng đường, thể lực."""
from dataclasses import dataclass, field

import numpy as np

import court as C
from players import L_WRIST, R_WRIST

SHUTTLE_K = 9.81 / (6.7 * 6.7)  # lực cản cầu lông: vận tốc rơi tới hạn ~6.7 m/s


# ---------------------------------------------------------------- quả cầu
def _as_cands(v):
    """Kết quả mỗi khung: bản cũ là (x, y) hoặc None, bản mới là danh sách (x, y, độ tin)."""
    if v is None:
        return []
    if len(v) and np.ndim(v[0]) == 0:
        return [(float(v[0]), float(v[1]))]
    return [(float(c[0]), float(c[1])) for c in v]


def clean_shuttle(raw, n_frames, frame_h):
    """raw: {frame: ứng viên} → mảng (N, 2) đã lọc nhiễu, NaN = không thấy.

    Ghép các ứng viên thành từng đoạn bay liền mạch; bỏ đoạn đứng yên / trôi chậm (cầu nằm dưới sàn,
    cầm trên tay, đèn), đoạn quá ngắn (nhận nhầm chập chờn); mỗi khung giữ điểm thuộc đoạn dài nhất.
    """
    thr = 0.09 * frame_h  # bước nhảy tối đa mỗi khung
    pts = [(f, x, y) for f in sorted(raw) if f < n_frames for x, y in _as_cands(raw[f])]
    segs = []      # mỗi đoạn: list (frame, x, y)
    alive = []
    for f, x, y in pts:
        alive = [sg for sg in alive if f - sg[-1][0] <= 4]
        best, bd = None, None
        for sg in alive:
            g = f - sg[-1][0]
            if g <= 0:
                continue
            d = np.hypot(x - sg[-1][1], y - sg[-1][2])
            if d <= thr * g and (bd is None or d < bd):
                best, bd = sg, d
        if best is None:
            best = [(f, x, y)]
            segs.append(best)
            alive.append(best)
        else:
            best.append((f, x, y))

    p = np.full((n_frames, 2), np.nan)
    owner = np.zeros(n_frames)
    for sg in segs:
        q = np.array([(x, y) for _, x, y in sg])
        span = sg[-1][0] - sg[0][0] + 1
        extent = np.hypot(*(q.max(0) - q.min(0)))
        # Quãng di chuyển thật (bỏ rung nhẹ): cầu đang bay đi vài chục px trong vài khung.
        if len(sg) < 4 or extent < 0.012 * frame_h or (span >= 10 and extent < 1.2 * span):
            continue
        # Đoạn dài có những khúc đứng yên lâu (cầu rơi xuống sàn rồi nằm đó) → cắt bỏ khúc đó.
        keep = np.ones(len(sg), bool)
        r0 = 0
        for k in range(1, len(sg) + 1):
            if k == len(sg) or np.hypot(*(q[k] - q[r0])) > 4:
                if sg[k - 1][0] - sg[r0][0] >= 10:
                    keep[r0:k] = False
                r0 = k
        score = int(keep.sum())
        for (f, x, y), kp in zip(sg, keep):
            if kp and score > owner[f]:
                p[f] = (x, y)
                owner[f] = score

    # Lấp khoảng trống ngắn (≤ 4 khung) bằng nội suy, không nối hai vật khác nhau.
    idx = np.flatnonzero(~np.isnan(p[:, 0]))
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 5 and np.hypot(*(p[b] - p[a])) <= thr * (b - a):
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
    # Làm mượt thêm ~0.6 s rồi chỉ cộng quãng đường theo từng bước 0.4 s: rung chân vài pixel
    # (ở nửa sân xa, 1 pixel có thể là 4–5 cm) không bị cộng dồn thành quãng đường ảo.
    win = max(1, int(fps * 0.6)) | 1
    pos = s.pos.copy()
    for d in range(2):
        col = pos[:, d]
        ok = ~np.isnan(col)
        if ok.sum() > win:
            num = np.convolve(np.where(ok, col, 0), np.ones(win), "same")
            den = np.convolve(ok.astype(float), np.ones(win), "same")
            pos[:, d] = np.where(ok, num / np.maximum(den, 1), np.nan)
    stride = max(1, int(fps * 0.4))
    step = np.zeros(n)
    idx = np.arange(0, n, stride)
    seg = np.hypot(*np.diff(pos[idx], axis=0).T)
    ok = ~np.isnan(seg) & (seg / (stride / fps) < 8.0)  # bỏ bước nhảy > 8 m/s (nhận nhầm)
    for k in np.flatnonzero(ok):
        step[idx[k] + 1:idx[k + 1] + 1] = seg[k] / stride  # rải đều trong khoảng để HUD tăng mượt
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
        sides = _side_candidates(shuttle[t], players, t)
        if not sides:
            continue
        cands.append((t, score, sides))

    # Lúc chạm vợt quả cầu hay bị người che / nhoè → quỹ đạo đứt vài khung đúng chỗ đổi hướng.
    # So hướng bay ngay trước và ngay sau đoạn mất dấu ngắn.
    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:]):
        if not (2 <= b - a <= int(0.5 * fps)) or a < 3 or b + 3 >= n:
            continue
        before, after = shuttle[a - 3:a + 1], shuttle[b:b + 4]
        if np.isnan(before).any() or np.isnan(after).any():
            continue
        v1, v2 = before[-1] - before[0], after[-1] - after[0]
        s1, s2 = np.hypot(*v1), np.hypot(*v2)
        if s1 < 2 or s2 < 3:
            continue
        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (s1 * s2), -1, 1)))
        if angle < 60:
            continue
        t = (a + b) // 2
        sides = _side_candidates((shuttle[a] + shuttle[b]) / 2, players, t)
        if sides:
            cands.append((t, angle / 60, sides))

    # Một cú đánh có thể sinh nhiều ứng viên liền nhau → giữ cái mạnh nhất trong 8 khung.
    cands.sort(key=lambda c: -c[1])
    taken = []
    for c in cands:
        if all(abs(c[0] - u[0]) > 8 for u in taken):
            taken.append(c)
    taken.sort(key=lambda c: c[0])

    # Chia thành các chuỗi liên tục (cách nhau ≤ 3 s), rồi chọn bên đánh theo luật luân phiên.
    hits, group = [], []
    for c in taken + [None]:
        if group and (c is None or (c[0] - group[-1][0]) / fps > 3.0):
            hits += _assign_sides(group, fps)
            group = []
        if c is not None:
            group.append(c)

    if onsets is not None and len(onsets):
        for h in hits:
            k = np.abs(onsets * fps - h.frame).argmin()
            if abs(onsets[k] * fps - h.frame) <= 4:
                h.frame = int(round(onsets[k] * fps))  # khớp đúng khung có tiếng vợt
    return hits


def _side_candidates(pt, players, t):
    """Với mỗi phía sân: (người gần quả cầu nhất, khoảng cách chuẩn hoá theo chiều cao người).

    Chuẩn hoá theo chiều cao để người đứng gần camera (rất to trong hình) không 'hút' hết các cú đánh
    của người phía xa có quả cầu nằm đè lên khung hình của mình.
    """
    best = {}
    for p in players:
        if not p.seen[t]:
            continue
        x1, y1, x2, y2 = p.box[t]
        w, h = x2 - x1, y2 - y1
        if h < 1 or not (x1 - 0.8 * w <= pt[0] <= x2 + 0.8 * w and y1 - 0.7 * h <= pt[1] <= y2 - 0.25 * h):
            continue
        wr = p.wrist[t]
        d = min(np.hypot(*(pt - wr[0])), np.hypot(*(pt - wr[1])), np.hypot(pt[0] - (x1 + x2) / 2, pt[1] - y1)) / h
        if d < best.get(p.side, (None, 9))[1]:
            best[p.side] = (p.pid, d)
    return best


def _assign_sides(group, fps, skip_penalty=1.1):
    """Viterbi: chọn phía đánh cho cả chuỗi sao cho hai bên luân phiên. Cú không hợp lý thì bỏ (nhận nhầm).

    Hai cú liên tiếp cùng phía nghĩa là đã sót một cú của bên kia ở giữa: chấp nhận dễ nếu hai cú cách nhau
    đủ lâu để có một lượt bay qua lại (> 1.6 s), gần như không chấp nhận nếu quá sát (< 0.8 s).
    """
    INF = 1e9

    def cost(c, s):
        return min(c[2][s][1], 1.5) if s in c[2] else INF

    def same_side_penalty(dt):
        return 0.2 if dt > 1.6 else (0.8 if dt > 0.8 else 2.5)

    # trạng thái: phía của cú được giữ gần nhất (0/1) hoặc -1 (chưa giữ cú nào); lưu cả khung của cú đó
    best = {-1: (0.0, [], None)}
    for k, c in enumerate(group):
        nxt = {}
        for last, (acc, path, last_f) in best.items():
            cand = (acc + skip_penalty, path, last_f)  # bỏ cú này
            if cand[0] < nxt.get(last, (INF,))[0]:
                nxt[last] = cand
            for s in (0, 1):
                e = cost(c, s)
                if e >= INF:
                    continue
                tr = same_side_penalty((c[0] - last_f) / fps) if last == s else 0.0
                cand = (acc + e + tr, path + [(k, s)], c[0])
                if cand[0] < nxt.get(s, (INF,))[0]:
                    nxt[s] = cand
        best = nxt
    _, path, _ = min(best.values(), key=lambda v: v[0])
    return [Hit(int(group[k][0]), int(group[k][2][s][0]), s) for k, s in path]


def estimate_speeds(hits, shuttle, players, courts, fps):
    """courts.at(khung) → Court tại khung đó (máy quay có thể xê dịch)."""
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
                ppm = courts.at(h.frame).px_per_meter(p.pos[h.frame])
                if ppm > 1:
                    v_img = float(np.median(lens)) * fps / ppm  # m/s, chỉ phần song song mặt ảnh
        v_flight = 0.0
        if k + 1 < len(hits):
            nxt = hits[k + 1]
            q = by_pid.get(nxt.player)
            dt = (nxt.frame - h.frame) / fps
            if q is not None and nxt.side != h.side and 0.3 <= dt < 3 and not np.isnan(q.pos[nxt.frame]).any():
                x = np.hypot(*(q.pos[nxt.frame] - p.pos[h.frame]))
                if 2 <= x <= 14:  # ngoài khoảng này là vị trí người bị nhiễu
                    v_flight = (np.exp(SHUTTLE_K * x) - 1) / (SHUTTLE_K * dt)
        # Ưu tiên ước tính theo thời gian bay (bền hơn); số đo trên ảnh chỉ dùng khi lệch không quá xa.
        if v_flight and v_img:
            v = max(v_flight, min(v_img * 1.15, v_flight * 1.6))
        else:
            v = v_flight or v_img * 1.15
        # Trên ~300 km/h là sai số vị trí (người phía xa bị ép mỏng), không phải cú đánh thật → để trống
        # thay vì ghi một con số tối đa giả.
        if 0 < v * 3.6 <= 300:
            h.speed_kmh = float(max(15.0, v * 3.6))


def build_rallies(hits, shuttle, courts, fps, gap_s=3.0, players=None):
    # Người chơi vẫn chạy mạnh trong khoảng trống → vẫn đang đánh, chỉ là mất dấu quả cầu vài giây.
    spd = np.nan_to_num(np.array([p.speed for p in players])) if players else None

    def still_playing(a, b):
        if spd is None or (b - a) / fps > 5.0:
            return False
        return float(np.mean(spd[:, a:b].max(0) > 2.0)) >= 0.3

    rallies, cur = [], []
    for h in hits:
        if cur and (h.frame - cur[-1].frame) / fps > gap_s and not still_playing(cur[-1].frame, h.frame):
            rallies.append(cur)
            cur = []
        cur.append(h)
    if cur:
        rallies.append(cur)
    # "Rally" chỉ 1 cú thường là nhiễu (nhặt cầu, đánh trả cầu cho sân bên) → bỏ.
    rallies = [hs for hs in rallies if len(hs) >= 2]
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
            lx, ly = courts.at(last).to_court([land])[0]
            r.land_court = (float(lx), float(ly))
            r.land_in = bool(-0.1 <= lx <= C.W + 0.1 and -0.1 <= ly <= C.L + 0.1)
        out.append(r)
    return out
