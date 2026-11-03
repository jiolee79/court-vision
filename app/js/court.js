// Kích thước sân cầu lông tiêu chuẩn (mét). Gốc toạ độ: góc trái của đường biên cuối phía XA camera.
// x: ngang sân (0 → 6.1), y: dọc sân (0 = biên cuối xa, 13.4 = biên cuối gần).
export const COURT = {
  W: 6.1,
  L: 13.4,
  NET: 6.7,
  SHORT: 1.98,      // vạch giao cầu ngắn cách lưới
  LONG_D: 0.76,     // vạch giao cầu dài (đánh đôi) cách biên cuối
  SINGLE_IN: 0.46,  // biên dọc đánh đơn cách biên dọc đánh đôi
};

// Các đoạn thẳng của sân trong toạ độ mét, dùng để vẽ preview hiệu chỉnh và bản đồ sân.
export function courtLines() {
  const { W, L, NET, SHORT, LONG_D, SINGLE_IN } = COURT;
  const s = [];
  const seg = (x1, y1, x2, y2) => s.push([x1, y1, x2, y2]);
  seg(0, 0, W, 0); seg(W, 0, W, L); seg(W, L, 0, L); seg(0, L, 0, 0);
  seg(SINGLE_IN, 0, SINGLE_IN, L); seg(W - SINGLE_IN, 0, W - SINGLE_IN, L);
  seg(0, NET - SHORT, W, NET - SHORT); seg(0, NET + SHORT, W, NET + SHORT);
  seg(0, LONG_D, W, LONG_D); seg(0, L - LONG_D, W, L - LONG_D);
  seg(W / 2, 0, W / 2, NET - SHORT); seg(W / 2, NET + SHORT, W / 2, L);
  return s;
}

export const NET_LINE = [0, COURT.NET, COURT.W, COURT.NET];
