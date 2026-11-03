"""Xuất thống kê ra Excel."""
import numpy as np
import pandas as pd


def _t(frame, fps, offset):
    s = offset + frame / fps
    return f"{int(s // 60)}:{s % 60:05.2f}"


def write_excel(path, players, rallies, hits, fps, offset_s=0.0):
    names = {p.pid: p.name for p in players}
    side_name = {0: "Đội gần", 1: "Đội xa"}

    rows = []
    for p in players:
        mine = [h for h in hits if h.player == p.pid]
        speeds = [h.speed_kmh for h in mine if h.speed_kmh]
        rows.append({
            "Người chơi": p.name,
            "Đội": side_name[p.side],
            "Quãng đường (m)": round(float(p.dist[-1]), 1),
            "Tốc độ chạy max (km/h)": round(float(np.nanmax(p.speed)) * 3.6, 1) if len(p.speed) else 0,
            "Số cú đánh": len(mine),
            "Cầu nhanh nhất (km/h)": round(max(speeds)) if speeds else None,
            "Tốc độ cầu TB (km/h)": round(float(np.mean(speeds))) if speeds else None,
            "Thể lực cuối trận": round(float(p.stamina[-1])),
            "% thời gian nhận diện được": round(100 * float(p.seen.mean()), 1),
        })
    summary = pd.DataFrame(rows)

    rally_rows = []
    for i, r in enumerate(rallies, 1):
        sp = [h.speed_kmh for h in r.hits if h.speed_kmh]
        last = r.hits[-1]
        rally_rows.append({
            "Rally": i,
            "Bắt đầu": _t(r.start, fps, offset_s),
            "Kết thúc": _t(r.end, fps, offset_s),
            "Thời gian (s)": round((r.end - r.start) / fps, 2),
            "Số cú": len(r.hits),
            "Cầu nhanh nhất (km/h)": round(max(sp)) if sp else None,
            "Người đánh cuối": names.get(last.player, "?"),
            "Cầu rơi (x, y m)": f"{r.land_court[0]:.1f}, {r.land_court[1]:.1f}" if r.land_court else "",
            "Trong sân (ước tính)": {True: "Trong", False: "Ngoài", None: ""}[r.land_in],
        })
    rally_df = pd.DataFrame(rally_rows)

    hit_rows = []
    for r_i, r in enumerate(rallies, 1):
        for k, h in enumerate(r.hits, 1):
            hit_rows.append({
                "Rally": r_i, "Cú thứ": k, "Thời điểm": _t(h.frame, fps, offset_s),
                "Người đánh": names.get(h.player, "?"), "Đội": side_name.get(h.side, ""),
                "Tốc độ cầu (km/h)": round(h.speed_kmh) if h.speed_kmh else None,
            })
    hit_df = pd.DataFrame(hit_rows)

    overview = pd.DataFrame([
        {"Chỉ số": "Số rally", "Giá trị": len(rallies)},
        {"Chỉ số": "Tổng số cú đánh", "Giá trị": len(hits)},
        {"Chỉ số": "Rally dài nhất (cú)", "Giá trị": max((len(r.hits) for r in rallies), default=0)},
        {"Chỉ số": "Rally dài nhất (giây)", "Giá trị": round(max(((r.end - r.start) / fps for r in rallies), default=0), 1)},
        {"Chỉ số": "Trung bình cú/rally", "Giá trị": round(np.mean([len(r.hits) for r in rallies]), 1) if rallies else 0},
        {"Chỉ số": "Cầu nhanh nhất (km/h)", "Giá trị": round(max((h.speed_kmh for h in hits), default=0))},
    ])

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        overview.to_excel(xw, sheet_name="Tổng quan", index=False)
        summary.to_excel(xw, sheet_name="Người chơi", index=False)
        rally_df.to_excel(xw, sheet_name="Rally", index=False)
        hit_df.to_excel(xw, sheet_name="Cú đánh", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col) + 2
                ws.column_dimensions[col[0].column_letter].width = min(40, max(10, width))
            ws.freeze_panes = "A2"
    return overview, summary
