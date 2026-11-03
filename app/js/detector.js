// Nhận diện người bằng MediaPipe Object Detector (EfficientDet-Lite), chạy hoàn toàn trên máy.
const MP_VERSION = '0.10.14';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;

export const MODELS = {
  fast: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
  accurate: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite',
};

export class PersonDetector {
  constructor() {
    this.detector = null;
    this.delegate = null;
    this.lastTs = 0;
    // Khung cắt vùng sân: phóng to vùng sân trước khi đưa vào model để người ở xa vẫn đủ lớn.
    this.crop = document.createElement('canvas');
    this.cropCtx = this.crop.getContext('2d');
  }

  async load(quality = 'fast', onStatus = () => {}) {
    onStatus('Đang tải thư viện AI…');
    const { ObjectDetector, FilesetResolver } = await import(`${MP_BASE}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: MODELS[quality] || MODELS.fast, delegate },
      runningMode: 'VIDEO',
      scoreThreshold: 0.3,
      maxResults: 12,
      categoryAllowlist: ['person'],
    });
    onStatus('Đang tải mô hình nhận diện…');
    try {
      // Một số máy treo khi khởi tạo GPU: quá 15 giây thì chuyển sang CPU.
      this.detector = await Promise.race([
        ObjectDetector.createFromOptions(fileset, opts('GPU')),
        new Promise((_, rej) => setTimeout(() => rej(new Error('GPU timeout')), 15000)),
      ]);
      this.delegate = 'GPU';
    } catch (e) {
      console.warn('GPU không khả dụng, chuyển sang CPU', e);
      this.detector = await ObjectDetector.createFromOptions(fileset, opts('CPU'));
      this.delegate = 'CPU';
    }
    onStatus('');
  }

  // source: canvas/video; roi: [x, y, w, h] trong toạ độ source. Trả về bbox trong toạ độ source.
  detect(source, roi) {
    if (!this.detector) return [];
    const [rx, ry, rw, rh] = roi;
    const maxSide = 640;
    const k = Math.min(1, maxSide / Math.max(rw, rh));
    const cw = Math.max(1, Math.round(rw * k));
    const ch = Math.max(1, Math.round(rh * k));
    if (this.crop.width !== cw || this.crop.height !== ch) {
      this.crop.width = cw;
      this.crop.height = ch;
    }
    this.cropCtx.drawImage(source, rx, ry, rw, rh, 0, 0, cw, ch);
    let ts = performance.now();
    if (ts <= this.lastTs) ts = this.lastTs + 1;
    this.lastTs = ts;
    const res = this.detector.detectForVideo(this.crop, ts);
    return res.detections.map((d) => {
      const b = d.boundingBox;
      return {
        x: rx + b.originX / k,
        y: ry + b.originY / k,
        w: b.width / k,
        h: b.height / k,
        score: d.categories[0]?.score ?? 0,
      };
    });
  }

  close() {
    this.detector?.close();
    this.detector = null;
  }
}
