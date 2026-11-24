const player = document.querySelector('.player');
const video = document.querySelector('#replay');
const start = document.querySelector('.player-start');
const quality = document.querySelector('#video-quality');
const status = document.querySelector('#playback-status');
const retry = document.querySelector('#retry-video');
const storageKey = 'go2-replay:' + player.dataset.videoUrl;
let position = 0;
let lastSaved = 0;
let pendingPosition = null;
let stallTimer;

try {
  position = Number(localStorage.getItem(storageKey)) || 0;
} catch {
  // Playback also works when browser storage is disabled.
}

function savePosition() {
  try {
    localStorage.setItem(storageKey, String(position));
  } catch {
    // Only the resume timestamp is persisted.
  }
}

function showStatus(message, canRetry = false) {
  status.textContent = message;
  status.hidden = !message;
  retry.hidden = !canRetry;
}

function loadVideo(play) {
  clearTimeout(stallTimer);
  start.hidden = true;
  pendingPosition = position;
  video.src = quality?.value || player.dataset.videoUrl;
  video.load();
  showStatus('正在连接 ICC…');
  stallTimer = setTimeout(() => showStatus('连接耗时较长。', true), 8000);
  if (play) {
    video.play().catch(error => {
      if (error.name === 'NotAllowedError') {
        showStatus('点击播放器里的播放按钮继续。');
      }
    });
  }
}

start.addEventListener('click', () => loadVideo(true));
retry.addEventListener('click', () => loadVideo(true));
quality?.addEventListener('change', () => {
  if (video.hasAttribute('src')) loadVideo(!video.paused);
});

video.addEventListener('loadedmetadata', () => {
  if (pendingPosition > 0 && pendingPosition < video.duration) video.currentTime = pendingPosition;
  pendingPosition = null;
});
video.addEventListener('timeupdate', () => {
  if (video.readyState < HTMLMediaElement.HAVE_METADATA || pendingPosition !== null) return;
  position = video.currentTime;
  if (Math.abs(position - lastSaved) >= 3) {
    savePosition();
    lastSaved = position;
  }
});
video.addEventListener('seeked', savePosition);
video.addEventListener('pause', savePosition);
window.addEventListener('pagehide', savePosition);
video.addEventListener('ended', () => {
  position = 0;
  savePosition();
});
for (const event of ['playing', 'canplay']) {
  video.addEventListener(event, () => {
    clearTimeout(stallTimer);
    showStatus('');
  });
}
video.addEventListener('waiting', () => {
  clearTimeout(stallTimer);
  showStatus('正在缓冲…');
  stallTimer = setTimeout(() => showStatus('缓冲耗时较长。', true), 8000);
});
video.addEventListener('error', () => {
  clearTimeout(stallTimer);
  showStatus('回放加载失败。SSH 隧道恢复后可以续播。', true);
});
