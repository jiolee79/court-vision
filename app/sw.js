// Service worker: lưu app + thư viện AI + mô hình vào bộ nhớ đệm để mở nhanh và dùng được khi mất mạng.
const VERSION = 'cv-v1';
const SHELL = [
  './',
  'index.html',
  'css/style.css',
  'js/main.js',
  'js/court.js',
  'js/homography.js',
  'js/detector.js',
  'js/tracker.js',
  'js/hits.js',
  'js/onset-core.js',
  'js/onset-worklet.js',
  'js/hud.js',
  'js/recorder.js',
  'manifest.webmanifest',
  'icons/icon.svg',
  'icons/icon-192.png',
  'icons/icon-512.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  const isLib = url.hostname === 'cdn.jsdelivr.net' || url.hostname === 'storage.googleapis.com';
  if (url.origin === location.origin) {
    // App: ưu tiên mạng để luôn có bản mới, mất mạng thì dùng bản đã lưu.
    e.respondWith(
      fetch(req)
        .then((res) => { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); return res; })
        .catch(() => caches.match(req, { ignoreSearch: true })),
    );
  } else if (isLib) {
    // Thư viện/mô hình có phiên bản cố định: ưu tiên bộ nhớ đệm.
    e.respondWith(
      caches.match(req).then((hit) => hit || fetch(req).then((res) => {
        if (res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); }
        return res;
      })),
    );
  }
});
