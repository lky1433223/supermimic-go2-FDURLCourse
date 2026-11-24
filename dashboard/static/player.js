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
let hls = null;
let started = false;

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
  started = true;
  const useHls = player.dataset.hlsUrl && ['auto', '0', '1'].includes(quality.value);
  if (hls) {
    hls.destroy();
    hls = null;
  }
  if (useHls && Hls.isSupported()) {
    hls = new Hls({startLevel: 0, maxBufferLength: 20, backBufferLength: 10});
    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      hls.currentLevel = quality.value === 'auto' ? -1 : Number(quality.value);
    });
    hls.on(Hls.Events.ERROR, (_, data) => {
      if (data.fatal) showStatus('视频连接中断，可以重新连接。', true);
    });
    hls.loadSource(player.dataset.hlsUrl);
    hls.attachMedia(video);
  } else {
    video.src = useHls ? player.dataset.videoUrl : (quality?.value || player.dataset.videoUrl);
    video.load();
  }
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
  if (hls && ['auto', '0', '1'].includes(quality.value)) {
    hls.currentLevel = quality.value === 'auto' ? -1 : Number(quality.value);
  } else if (started) loadVideo(!video.paused);
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
