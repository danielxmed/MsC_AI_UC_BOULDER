'use strict';
const CACHE = 'dl-opt-42f9738e';
const PRECACHE_URLS = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "./media/apple-touch-icon.png",
  "./media/cover.png",
  "./media/file0.svg",
  "./media/file1.svg",
  "./media/file10.svg",
  "./media/file11.svg",
  "./media/file12.svg",
  "./media/file13.svg",
  "./media/file14.svg",
  "./media/file15.svg",
  "./media/file16.svg",
  "./media/file17.svg",
  "./media/file18.svg",
  "./media/file19.svg",
  "./media/file2.svg",
  "./media/file20.svg",
  "./media/file21.svg",
  "./media/file22.svg",
  "./media/file23.svg",
  "./media/file24.svg",
  "./media/file25.svg",
  "./media/file26.svg",
  "./media/file27.png",
  "./media/file28.png",
  "./media/file29.jpg",
  "./media/file3.png",
  "./media/file30.svg",
  "./media/file31.svg",
  "./media/file32.svg",
  "./media/file33.svg",
  "./media/file34.svg",
  "./media/file35.svg",
  "./media/file36.svg",
  "./media/file37.svg",
  "./media/file38.svg",
  "./media/file39.svg",
  "./media/file4.png",
  "./media/file40.svg",
  "./media/file41.svg",
  "./media/file42.svg",
  "./media/file43.png",
  "./media/file44.png",
  "./media/file5.svg",
  "./media/file6.svg",
  "./media/file7.svg",
  "./media/file8.svg",
  "./media/file9.svg",
  "./media/icon-192.png",
  "./media/icon-512.png"
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req).then((resp) => {
        if (!resp || resp.status !== 200 || resp.type !== 'basic') return resp;
        const copy = resp.clone();
        caches.open(CACHE).then((cache) => cache.put(req, copy));
        return resp;
      }).catch(() => {
        if (req.mode === 'navigate') return caches.match('./index.html');
      });
    })
  );
});
