"""Cửa sổ hiệu chỉnh sân: kéo 4 điểm mốc vào đúng vị trí, xem trước các đường sân vẽ lại.

Phím: Enter/Space = xong · M = đổi bộ điểm mốc · Z = thu nhỏ (điểm nằm ngoài khung) · R = đặt lại · Esc = huỷ
(Chữ trong cửa sổ OpenCV không hiển thị được dấu tiếng Việt nên ghi không dấu.)
"""
import cv2
import numpy as np

from court import NET, W, Court, court_lines

WIN = "Hieu chinh san - Court Vision"
REF_ORDER = ["corners", "net_short", "net_base"]
REF_ASCII = {
    "corners": ("4 goc san", ["1 Xa-trai", "2 Xa-phai", "3 Gan-phai", "4 Gan-trai"]),
    "net_short": ("2 chan cot luoi + vach giao cau ngan phia gan",
                  ["1 Cot luoi trai", "2 Cot luoi phai", "3 Vach ngan gan-phai", "4 Vach ngan gan-trai"]),
    "net_base": ("2 chan cot luoi + 2 goc bien cuoi phia gan",
                 ["1 Cot luoi trai", "2 Cot luoi phai", "3 Goc gan-phai", "4 Goc gan-trai"]),
}


def default_points(w, h):
    return np.float32([[w * .36, h * .3], [w * .64, h * .3], [w * .88, h * .88], [w * .12, h * .88]])


def draw_court(canvas, court, tr, thick):
    """Vẽ các đường sân; đoạn nằm sau lưng camera được bỏ (máy đặt sát sân)."""
    for x1, y1, x2, y2 in court_lines() + [(0, NET, W, NET)]:
        t = np.linspace(0, 1, 24)[:, None]
        seg = np.c_[x1 + (x2 - x1) * t[:, 0], y1 + (y2 - y1) * t[:, 0]]
        p = np.c_[seg, np.ones(len(seg))] @ court.Hinv.T
        ok = p[:, 2] > 1e-6
        pts = [tr(q[:2] / q[2]) for q, v in zip(p, ok) if v]
        net = y1 == y2 == NET and x1 == 0 and x2 == W
        for a, b in zip(pts[:-1], pts[1:]):
            if max(abs(a[0]), abs(a[1]), abs(b[0]), abs(b[1])) < 20000:
                cv2.line(canvas, a, b, (63, 210, 255) if net else (255, 208, 63), thick + (1 if net else 0), cv2.LINE_AA)


def calibrate(get_frame, t=0.0, duration=None, initial=None, ref="corners"):
    """get_frame(t) → khung hình tại giây t. Phím A/D lùi/tiến 5 giây để chọn khung nhìn rõ vạch sân.
    Trả về (Court, t của khung đã chọn), hoặc None nếu người dùng huỷ."""
    frame = get_frame(t)
    h, w = frame.shape[:2]
    disp_scale = min(1.0, 860 / h, 1500 / w)
    pts = np.float32(initial if initial is not None else default_points(w, h))
    st = {"zoom": bool((pts[:, 0] < 0).any() or (pts[:, 1] < 0).any() or (pts[:, 0] > w).any() or (pts[:, 1] > h).any()),
          "drag": -1, "start": None, "ref": ref}

    def view():
        k = 0.5 if st["zoom"] else 1.0
        return k, w * (1 - k) / 2, h * (1 - k) / 2

    def to_img(mx, my):
        k, ox, oy = view()
        return (mx / disp_scale - ox) / k, (my / disp_scale - oy) / k

    def on_mouse(ev, mx, my, flags, _):
        x, y = to_img(mx, my)
        if ev == cv2.EVENT_LBUTTONDOWN:
            d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            i = int(d.argmin())
            if d[i] < 60 / disp_scale / view()[0]:
                st["drag"], st["start"] = i, (x, y, *pts[i])
        elif ev == cv2.EVENT_MOUSEMOVE and st["drag"] >= 0:
            sx, sy, ox, oy = st["start"]
            pts[st["drag"]] = (np.clip(ox + x - sx, -w * .5, w * 1.5), np.clip(oy + y - sy, -h * .5, h * 1.5))
        elif ev == cv2.EVENT_LBUTTONUP:
            st["drag"] = -1

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        k, ox, oy = view()
        canvas = np.full_like(frame, 25)
        canvas = cv2.warpAffine(frame, np.float32([[k, 0, ox], [0, k, oy]]), (w, h), dst=canvas,
                                borderMode=cv2.BORDER_TRANSPARENT)

        def tr(p):
            return int(p[0] * k + ox), int(p[1] * k + oy)

        thick = max(1, int(2 / disp_scale))
        try:
            draw_court(canvas, Court(pts, st["ref"]), tr, thick)
        except cv2.error:
            pass
        name, labels = REF_ASCII[st["ref"]]
        for i, p in enumerate(pts):
            c = tr(p)
            cv2.circle(canvas, c, int(13 / disp_scale), (0, 255, 255), thick + 1, cv2.LINE_AA)
            cv2.putText(canvas, labels[i], (c[0] + int(14 / disp_scale), c[1] - int(10 / disp_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55 / disp_scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
            cv2.putText(canvas, labels[i], (c[0] + int(14 / disp_scale), c[1] - int(10 / disp_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55 / disp_scale, (0, 255, 255), thick, cv2.LINE_AA)
        if st["drag"] >= 0:  # kính lúp
            px, py = pts[st["drag"]]
            r = 70
            crop = cv2.getRectSubPix(frame, (r, r), (float(px), float(py)))
            crop = cv2.resize(crop, (r * 3, r * 3), interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(crop, (r * 3 // 2, r * 3 // 2), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            canvas[10:10 + r * 3, w - 10 - r * 3:w - 10] = crop
        out = cv2.resize(canvas, None, fx=disp_scale, fy=disp_scale, interpolation=cv2.INTER_AREA)
        lines = [f"Khung {int(t // 60)}:{t % 60:04.1f}  (A/D: lui/tien 5 giay)   Moc: {name}",
                 "Keo 4 diem vao dung vi tri. Enter: xong | M: doi bo moc | Z: thu nho | R: dat lai | Esc: huy"]
        cv2.rectangle(out, (0, 0), (out.shape[1], 46), (0, 0, 0), -1)
        for i, s in enumerate(lines):
            cv2.putText(out, s, (8, 18 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, out)
        key = cv2.waitKey(16) & 0xFF
        if key in (13, 32):
            cv2.destroyWindow(WIN)
            return Court(pts, st["ref"]), t
        if key in (ord("a"), ord("A"), ord("d"), ord("D")):
            t = max(0.0, t + (5 if key in (ord("d"), ord("D")) else -5))
            if duration:
                t = min(t, max(0.0, duration - 0.5))
            frame = get_frame(t)
        if key == 27 or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None
        if key in (ord("m"), ord("M")):
            st["ref"] = REF_ORDER[(REF_ORDER.index(st["ref"]) + 1) % len(REF_ORDER)]
        if key in (ord("z"), ord("Z")):
            st["zoom"] = not st["zoom"]
        if key in (ord("r"), ord("R")):
            pts[:] = default_points(w, h)
