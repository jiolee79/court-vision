"""Cửa sổ hiệu chỉnh sân: kéo 4 điểm vào 4 góc sân, xem trước các đường sân.

Phím: Enter/Space = xong · R = đặt lại · Z = thu nhỏ (khi góc sân ngoài khung) · Esc = huỷ
"""
import cv2
import numpy as np

from court import Court, court_lines, NET, W

WIN = "Hieu chinh san - keo 4 diem vao 4 goc san, Enter de xong"


def calibrate(frame, initial=None):
    h, w = frame.shape[:2]
    disp_scale = min(1.0, 900 / h, 1500 / w)
    corners = np.float32(initial if initial is not None else
                         [[w * .36, h * .3], [w * .64, h * .3], [w * .88, h * .88], [w * .12, h * .88]])
    state = {"zoom": any(x < 0 or y < 0 or x > w or y > h for x, y in corners), "drag": -1, "start": None}

    def view_params():
        k = 0.5 if state["zoom"] else 1.0
        return k, w * (1 - k) / 2, h * (1 - k) / 2

    def to_img(mx, my):
        k, ox, oy = view_params()
        return (mx / disp_scale - ox) / k, (my / disp_scale - oy) / k

    def on_mouse(ev, mx, my, flags, _):
        x, y = to_img(mx, my)
        if ev == cv2.EVENT_LBUTTONDOWN:
            d = np.hypot(corners[:, 0] - x, corners[:, 1] - y)
            i = int(d.argmin())
            k = view_params()[0]
            if d[i] < 60 / disp_scale / k:
                state["drag"] = i
                state["start"] = (x, y, *corners[i])
        elif ev == cv2.EVENT_MOUSEMOVE and state["drag"] >= 0:
            sx, sy, ox, oy = state["start"]
            corners[state["drag"]] = (np.clip(ox + x - sx, -w * .5, w * 1.5), np.clip(oy + y - sy, -h * .5, h * 1.5))
        elif ev == cv2.EVENT_LBUTTONUP:
            state["drag"] = -1

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        k, ox, oy = view_params()
        canvas = np.full_like(frame, 20)
        M = np.float32([[k, 0, ox], [0, k, oy]])
        canvas = cv2.warpAffine(frame, M, (w, h), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
        tr = lambda p: (int(p[0] * k + ox), int(p[1] * k + oy))
        try:
            court = Court(corners)
            for x1, y1, x2, y2 in court_lines():
                a, b = court.to_image([[x1, y1], [x2, y2]])
                cv2.line(canvas, tr(a), tr(b), (255, 208, 63), 2, cv2.LINE_AA)
            a, b = court.to_image([[0, NET], [W, NET]])
            cv2.line(canvas, tr(a), tr(b), (63, 210, 255), 3, cv2.LINE_AA)
        except cv2.error:
            pass
        for i, p in enumerate(corners):
            c = tr(p)
            cv2.circle(canvas, c, int(14 / disp_scale), (255, 208, 63), 3, cv2.LINE_AA)
            cv2.putText(canvas, str(i + 1), (c[0] - 8, c[1] - int(20 / disp_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2 / disp_scale * .6, (255, 255, 255), 2, cv2.LINE_AA)
        if state["drag"] >= 0:  # kính lúp
            px, py = corners[state["drag"]]
            r = 80
            crop = cv2.getRectSubPix(frame, (r, r), (float(px), float(py)))
            crop = cv2.resize(crop, (r * 3, r * 3), interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(crop, (r * 3 // 2, r * 3 // 2), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            canvas[10:10 + r * 3, 10:10 + r * 3] = crop
        hint = "1=xa-trai 2=xa-phai 3=gan-phai 4=gan-trai | Enter: xong  Z: thu nho  R: dat lai  Esc: huy"
        out = cv2.resize(canvas, None, fx=disp_scale, fy=disp_scale, interpolation=cv2.INTER_AREA)
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(out, hint, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, out)
        key = cv2.waitKey(16) & 0xFF
        if key in (13, 32):
            cv2.destroyWindow(WIN)
            return Court(corners)
        if key == 27 or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None
        if key in (ord('z'), ord('Z')):
            state["zoom"] = not state["zoom"]
        if key in (ord('r'), ord('R')):
            corners[:] = [[w * .36, h * .3], [w * .64, h * .3], [w * .88, h * .88], [w * .12, h * .88]]
