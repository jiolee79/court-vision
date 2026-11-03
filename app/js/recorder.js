// Ghi lại canvas (video + HUD) kèm âm thanh thành file video.
const TYPES = [
  'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
  'video/mp4',
  'video/webm;codecs=vp9,opus',
  'video/webm;codecs=vp8,opus',
  'video/webm',
];

export class HudRecorder {
  constructor() {
    this.rec = null;
    this.chunks = [];
  }

  get recording() {
    return this.rec?.state === 'recording';
  }

  start(canvas, audioTrack) {
    const stream = canvas.captureStream(30);
    if (audioTrack) stream.addTrack(audioTrack);
    const mimeType = TYPES.find((t) => window.MediaRecorder?.isTypeSupported?.(t)) || '';
    this.rec = new MediaRecorder(stream, {
      mimeType,
      videoBitsPerSecond: 6_000_000,
    });
    this.chunks = [];
    this.rec.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.rec.start(1000);
    this.startedAt = Date.now();
  }

  stop() {
    return new Promise((resolve) => {
      if (!this.rec) return resolve(null);
      this.rec.onstop = () => {
        const type = this.rec.mimeType || 'video/webm';
        const blob = new Blob(this.chunks, { type });
        this.rec.stream.getVideoTracks().forEach((t) => t.stop());
        this.rec = null;
        resolve(blob);
      };
      this.rec.stop();
    });
  }
}

// Lưu/chia sẻ file: trên điện thoại ưu tiên bảng chia sẻ (lưu vào Ảnh), không có thì tải về.
export async function saveBlob(blob, baseName) {
  const ext = blob.type.includes('mp4') ? 'mp4' : blob.type.includes('webm') ? 'webm' : 'txt';
  const file = new File([blob], `${baseName}.${ext}`, { type: blob.type });
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: baseName });
      return;
    } catch (e) {
      if (e.name === 'AbortError') return;
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = file.name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
