"""Cửa sổ chọn video + nhập tên người chơi, rồi chạy phân tích (dành cho người không dùng dòng lệnh)."""
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

SETTINGS = Path.home() / "court-vision-ai" / "launcher.json"


def load_settings():
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main():
    root = tk.Tk()
    root.title("Court Vision – Phân tích video cầu lông")
    root.resizable(False, False)
    st = load_settings()
    pad = {"padx": 10, "pady": 4}

    video = tk.StringVar(value=st.get("video", ""))
    singles = tk.BooleanVar(value=st.get("singles", False))
    quality = tk.StringVar(value=st.get("quality", "fast"))
    recal = tk.BooleanVar(value=False)
    names = [tk.StringVar(value=v) for v in (st.get("names") or ["", "", "", ""])]
    start = tk.StringVar(value="0")
    end = tk.StringVar(value="0")

    def pick():
        f = filedialog.askopenfilename(title="Chọn video trận đấu",
                                       filetypes=[("Video", "*.mp4 *.mov *.m4v *.avi *.mkv"), ("Tất cả", "*.*")])
        if f:
            video.set(f)

    frm = ttk.Frame(root, padding=10)
    frm.grid()
    ttk.Label(frm, text="Video:").grid(row=0, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=video, width=48).grid(row=0, column=1, columnspan=2, **pad)
    ttk.Button(frm, text="Chọn…", command=pick).grid(row=0, column=3, **pad)

    ttk.Checkbutton(frm, text="Đánh đơn (2 người)", variable=singles).grid(row=1, column=1, sticky="w", **pad)
    labels = ["Gần camera – trái", "Gần camera – phải", "Phía xa – trái", "Phía xa – phải"]
    for i, lab in enumerate(labels):
        ttk.Label(frm, text=lab + ":").grid(row=2 + i, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=names[i], width=24).grid(row=2 + i, column=1, sticky="w", **pad)
    ttk.Label(frm, text="(Đánh đơn: dùng ô 1 và ô 3)").grid(row=2, column=2, columnspan=2, sticky="w")

    ttk.Label(frm, text="Chất lượng:").grid(row=6, column=0, sticky="w", **pad)
    q = ttk.Frame(frm)
    q.grid(row=6, column=1, columnspan=3, sticky="w")
    ttk.Radiobutton(q, text="Nhanh", value="fast", variable=quality).pack(side="left")
    ttk.Radiobutton(q, text="Chính xác nhất (chậm hơn ~2 lần)", value="best", variable=quality).pack(side="left", padx=10)

    ttk.Label(frm, text="Chỉ phân tích đoạn (giây):").grid(row=7, column=0, sticky="w", **pad)
    r = ttk.Frame(frm)
    r.grid(row=7, column=1, columnspan=3, sticky="w")
    ttk.Entry(r, textvariable=start, width=6).pack(side="left")
    ttk.Label(r, text=" đến ").pack(side="left")
    ttk.Entry(r, textvariable=end, width=6).pack(side="left")
    ttk.Label(r, text="  (0 đến 0 = cả video)").pack(side="left")
    ttk.Checkbutton(frm, text="Hiệu chỉnh lại sân", variable=recal).grid(row=8, column=1, sticky="w", **pad)

    result = {}

    def go():
        if not Path(video.get()).is_file():
            messagebox.showerror("Court Vision", "Hãy chọn file video.")
            return
        try:
            s, e = float(start.get() or 0), float(end.get() or 0)
        except ValueError:
            messagebox.showerror("Court Vision", "Thời gian bắt đầu/kết thúc phải là số giây.")
            return
        nm = [v.get().strip() for v in names]
        if singles.get():
            nm = [nm[0], nm[2]]
        result.update(video=video.get(), names=",".join(x for x in nm), singles=singles.get(),
                      quality=quality.get(), start=s, end=e, recal=recal.get())
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps({"video": video.get(), "singles": singles.get(), "quality": quality.get(),
                                        "names": [v.get() for v in names]}, ensure_ascii=False), encoding="utf-8")
        root.destroy()

    ttk.Button(frm, text="▶  Bắt đầu phân tích", command=go).grid(row=9, column=1, columnspan=2, pady=12)
    root.mainloop()
    if not result:
        return

    argv = [result["video"], "--names", result["names"], "--quality", result["quality"],
            "--start", str(result["start"]), "--end", str(result["end"])]
    if result["singles"]:
        argv.append("--singles")
    if result["recal"]:
        argv.append("--recalibrate")
    sys.argv = ["analyze.py"] + argv
    import analyze
    analyze.main()
    out_dir = Path(result["video"]).parent
    print(f"\nKết quả nằm trong thư mục: {out_dir}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
