# Court Vision – Thống kê cầu lông bằng AI

Gồm hai phần dùng kết hợp:

| | App điện thoại (`app/`) | Công cụ máy tính (`analyzer/`) |
|---|---|---|
| Dùng khi | Xem nhanh ngay tại sân | Muốn kết quả chính xác, video HUD hoàn chỉnh |
| Quả cầu | Không theo dõi (đếm cú đánh bằng âm thanh) | **TrackNetV3** – AI chuyên theo dõi cầu lông |
| Người chơi | MediaPipe + màu áo | **YOLO11-pose** (khung xương) + màu áo |
| Kết quả | HUD trực tiếp, ghi video | Video MP4 có HUD + file Excel thống kê |

## Công cụ máy tính (chính xác nhất)

**Cài đặt (một lần):** bấm đúp `Cai dat.bat`. Phiên bản thư viện được cố định trong `analyzer/requirements.txt` (bộ đã chạy thử). Thư viện AI (~1.5 GB) được cài vào
`C:\Users\<bạn>\court-vision-ai` (ngoài Dropbox).

**Dùng:** bấm đúp `Phan tich video.bat`
1. Chọn video, nhập tên 4 người (gần camera trái/phải, phía xa trái/phải), bấm *Bắt đầu*.
2. Lần đầu với mỗi video: cửa sổ hiệu chỉnh hiện ra → kéo 4 điểm mốc vào đúng chỗ → **Enter**.
   - **A / D**: lùi/tiến 5 giây để chọn khung nhìn rõ vạch sân nhất.
   - **M**: đổi bộ điểm mốc — *4 góc sân*, *2 chân cột lưới + 2 đầu vạch giao cầu ngắn phía gần*,
     hoặc *2 chân cột lưới + 2 góc biên cuối phía gần* (dùng khi biên cuối nằm ngoài khung).
   - **Z**: thu nhỏ để đặt điểm nằm ngoài khung hình.
   - Chỉ cần hiệu chỉnh **một lần** kể cả khi máy quay bị xê dịch giữa chừng: công cụ tự đo chuyển động
     máy (dựa vào trần/tường) và dời sân theo. Hiệu chỉnh được lưu cạnh video (`.court.json`).
3. Chờ xử lý. Kết quả nằm **cạnh file video**: `…_HUD.mp4` và `…_thong-ke.xlsx`.

Dòng lệnh: `python analyzer/analyze.py video.mp4 --names "A,B,C,D" [--singles] [--start 60 --end 180] [--quality best]`

**Thời gian xử lý** (CPU Intel i7-12700, không cần card đồ hoạ): video quay dọc 7 phút ≈ 12–17 phút
(nhận diện ~2/3, xuất video ~1/3). Nhanh hơn khi:
- quay **ngang** (quét cầu 1 ô thay vì 2);
- trận có nhiều lúc nghỉ — đoạn không có tiếng vợt > 4 giây được bỏ qua khi quét cầu;
- chỉ phân tích một đoạn (ô *Chỉ phân tích đoạn*) — kết quả và file lưu AI của đoạn có tên riêng (`…_100-125s_…`), không ghi đè kết quả cả video;
- chạy lại cùng video (đổi tên, xuất lại): dùng kết quả AI đã lưu (`.cvcache.pkl`), chỉ mất vài phút.

Tăng tốc đã dùng: OpenVINO + nén INT8 cho cả hai mô hình, giải mã và nén video bằng phần cứng Intel
Quick Sync, xuất video trên nhiều tiến trình song song. Lần chạy đầu tiên mất thêm ~1 phút để nén mô hình.

**Excel gồm:** Tổng quan · Người chơi (quãng đường, tốc độ chạy, số cú, cầu nhanh nhất) · Rally
(thời gian, số cú, người đánh cuối, cầu rơi trong/ngoài – ước tính) · Cú đánh (từng cú, ai đánh, tốc độ).

**Lọc nhiễu (sân đông người):** chỉ nhận người có chân nằm trong sân (lề 1 m hai bên, 2.6 m sau biên cuối),
bỏ người bị mép ảnh cắt mà đứng ngoài biên, người có kích thước không khớp với vị trí (người ở sân bên/khán đài
bị chiếu nhầm vào sân), và người đứng yên một chỗ ngoài sân gần như suốt video (trọng tài, người xem).
Quả cầu: AI giữ vài ứng viên mỗi khung, ghép thành các đường bay liền mạch rồi bỏ cầu nằm yên dưới sàn / cầm
trên tay và các vệt chập chờn (cầu sân bên). Rally chỉ 1 cú (nhặt cầu, đánh trả cầu) không tính.

**Cách tính:** cú đánh = điểm quỹ đạo cầu đổi hướng/tăng tốc đột ngột (kể cả khi cầu bị che vài khung
đúng lúc chạm vợt) cạnh một người chơi; bên đánh được chọn cho cả rally theo luật luân phiên (Viterbi),
khoảng cách cầu–cổ tay tính theo tỷ lệ chiều cao người để người đứng gần camera không "hút" cú đánh của
người phía xa. Tốc độ cầu = từ thời gian bay giữa hai người đánh (có tính lực cản) kết hợp đo trực tiếp
trên ảnh ngay sau cú đánh.

**Cách quay để số liệu chính xác nhất:** đặt điện thoại trên chân máy **cao 1.5–2.5 m**, **phía sau biên
cuối** (giữa sân), thấy được cả 4 góc sân, và **không chạm vào máy** trong lúc quay. Quay ngang nếu được.
Đặt máy sát sàn/cạnh lưới vẫn chạy được nhưng nửa sân xa bị ép rất mỏng (1 m chỉ còn vài chục pixel)
→ quãng đường và tốc độ của người phía xa kém chính xác.

Mô hình: TrackNetV3 (Chen & Wang, NYCU, MIT License) · YOLO11-pose (Ultralytics, AGPL-3.0).

---

## App điện thoại

Web app (PWA) cài được lên màn hình chính iPhone/Android như một app thường. AI chạy ngay trên máy
(MediaPipe, dùng GPU), không gửi video lên server. Link: https://jiolee79.github.io/court-vision/

## Tính năng
- **Hiệu chỉnh sân**: kéo 4 điểm mốc, có kính lúp khi kéo.
  - **📍 Đổi điểm mốc**: *4 góc sân*, *chân 2 cột lưới + 2 đầu vạch giao cầu ngắn phía gần* (khi không lùi
    đủ xa để thấy biên cuối gần — 4 điểm này luôn nằm giữa khung hình), hoặc *chân cột lưới + biên cuối gần*.
    Khi đổi, sân đang vẽ được giữ nguyên, chỉ cần tinh chỉnh.
  - **🔍 Thu nhỏ** (2× / 3.3× / 5×): kéo điểm ra ngoài video khi góc sân bị khuất. Video được đặt lệch lên trên
    để chừa nhiều chỗ phía dưới (biên cuối gần thường bị khuất ở đó).
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
