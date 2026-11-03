# Court Vision – Thống kê cầu lông bằng AI trên điện thoại

Web app (PWA) cài được lên màn hình chính iPhone/Android như một app thường. AI chạy ngay trên máy
(MediaPipe, dùng GPU), không gửi video lên server.

## Tính năng
- **Hiệu chỉnh sân**: kéo 4 điểm vào 4 góc sân, có kính lúp khi kéo. Nút **🔍 Góc ngoài khung** dùng khi một góc sân nằm ngoài khung hình.
- **Nhận diện & theo dõi 4 người** (hoặc 2 khi đánh đơn): tên nổi trên đầu, vòng sáng dưới chân, bản đồ sân nhìn từ trên xuống kèm đường di chuyển.
- **Quãng đường (MOVE)** tính bằng mét thật trên sân, **thể lực (STAMINA)** giảm khi chạy/bứt tốc và hồi khi đứng nghỉ.
- **Đếm cú đánh tự động** bằng tiếng "bốp" của cầu (micro / âm thanh video), có lọc theo chuyển động vung vợt để bỏ tiếng sân bên cạnh. Có thể chuyển sang **bấm tay**.
- **Rally time / Rally count / Shot speed** như HUD truyền hình.
- **Ghi lại video kèm HUD** (MP4 trên iPhone/Chrome mới, WebM nếu máy không hỗ trợ), lưu/chia sẻ thẳng từ điện thoại.
- **Tổng kết trận**: số rally, rally dài nhất, quãng đường, tốc độ chạy max, số cú đánh và cú cầu nhanh nhất của từng người.
- Chạy được khi không có mạng sau lần mở đầu tiên (thư viện AI và mô hình được lưu lại).

## Chạy thử trên máy tính
```bash
python serve.py
```
Mở http://localhost:8000 trong trình duyệt. Trên điện thoại cùng Wi-Fi, mở địa chỉ IP mà script in ra.
Qua `http://IP` chỉ dùng được chế độ **Phân tích video có sẵn**. Camera trực tiếp và cài app cần **https**.

## Đưa lên điện thoại (https, miễn phí)
Chọn một trong các cách sau, sau đó upload **nguyên thư mục `app/`**:
- **Netlify Drop**: vào app.netlify.com/drop rồi kéo thả thư mục `app`.
- **GitHub Pages**: đẩy thư mục `app` lên một repo, bật Pages.
- **Cloudflare Pages**: tạo project, upload thư mục `app`.

Mở link https trên điện thoại, rồi cài như app:
- iPhone (Safari): Chia sẻ ⬆️ → *Thêm vào MH chính*.
- Android (Chrome): menu ⋮ → *Cài đặt ứng dụng*.

## Cách quay để có kết quả tốt
- Đặt điện thoại trên chân máy, cao khoảng 1.5–2.5 m, phía sau biên cuối sân hoặc ở góc sân, sao cho thấy càng nhiều đường biên càng tốt.
- Cố định máy suốt trận. Nếu lỡ dịch máy, vào ⚙️ → *Hiệu chỉnh lại sân*.
- Nếu app gán nhầm tên giữa 2 người cùng đội, vào ⚙️ → *Đổi tên đội gần/xa*. Sau mỗi game, bấm *Hai đội đổi sân*.
- Sân bên cạnh ồn thì giảm *Độ nhạy*. Nếu bỏ sót cú đánh thì tăng lên.

## Các chỉ số được tính thế nào
| Chỉ số | Cách tính |
|---|---|
| Vị trí / MOVE | Điểm chân người (đáy khung nhận diện) chiếu xuống mặt sân bằng homography từ 4 góc đã hiệu chỉnh. |
| Cú đánh | Tiếng vợt chạm cầu (năng lượng âm cao tần tăng đột ngột), sau đó chọn người có chuyển động thân trên mạnh nhất lúc đó. Hai bên luân phiên đánh. |
| SHOT SPEED | **Ước tính** vận tốc ban đầu của cầu từ quãng đường giữa 2 người đánh liên tiếp và thời gian bay, có tính lực cản không khí của quả cầu (vận tốc rơi tới hạn ≈ 6.7 m/s). Không đo trực tiếp quả cầu nên có sai số, nhất là với cầu bổng. |
| STAMINA | Mô hình đơn giản theo quãng đường, mức bứt tốc và thời gian nghỉ. Không phải đo sinh lý thật. |

## Cấu trúc
```
app/
  index.html, css/style.css, manifest.webmanifest, sw.js, icons/
  js/main.js         – giao diện, vòng lặp xử lý, camera/video, ghi hình
  js/detector.js     – nhận diện người (MediaPipe EfficientDet), cắt vùng sân để người ở xa vẫn đủ lớn
  js/tracker.js      – gán ID người chơi, quãng đường, tốc độ, thể lực
  js/hits.js         – phát hiện cú đánh, rally, ước tính tốc độ cầu
  js/onset-core.js   – thuật toán phát hiện tiếng đánh (dùng trong AudioWorklet)
  js/hud.js          – vẽ HUD
  js/homography.js, js/court.js – toán phối cảnh và kích thước sân
serve.py             – server chạy thử
```

## Tối ưu cho điện thoại
- AI chỉ chạy trên vùng quanh sân (đã cắt và phóng to), không chạy trên cả khung hình.
- Tần suất nhận diện tự điều chỉnh theo sức mạnh máy, khoảng 4–20 lần/giây. Vị trí được làm mượt giữa các lần nhận diện.
- Ưu tiên GPU, tự chuyển sang CPU nếu máy không hỗ trợ. Có chế độ *Nhanh* (Lite0) và *Chính xác* (Lite2).
- Xử lý ở tối đa 1280 px, giữ màn hình luôn sáng khi đang quay.
