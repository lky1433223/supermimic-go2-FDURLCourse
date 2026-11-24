const rows = [...document.querySelectorAll('[data-evaluation]')];
const iterationFilter = document.querySelector('#iteration-filter');
const taskFilter = document.querySelector('#task-filter');
const videoFilter = document.querySelector('#video-filter');

function filterEvaluations() {
  for (const row of rows) {
    const matchesIteration = !iterationFilter.value || row.dataset.iteration === iterationFilter.value;
    const matchesTask = !taskFilter.value || row.dataset.task === taskFilter.value;
    const matchesVideo = !videoFilter.checked || Number(row.dataset.videos) > 0;
    row.hidden = !(matchesIteration && matchesTask && matchesVideo);
  }
  document.querySelector('#filter-empty').hidden = rows.some(row => !row.hidden);
}

if (rows.length) {
  for (const control of [iterationFilter, taskFilter, videoFilter]) {
    control.addEventListener('change', filterEvaluations);
  }
  filterEvaluations();
}

const chart = document.querySelector('#training-chart');
if (chart) {
  const select = document.querySelector('#metric-select');
  const series = JSON.parse(document.querySelector('#training-series').textContent);
  const stats = new Intl.NumberFormat('zh-CN', {maximumSignificantDigits: 4});

  function drawChart() {
    const points = series[select.value] || [];
    if (!points.length) return;
    const width = chart.clientWidth;
    const height = chart.clientHeight;
    if (width < 100 || height < 50) return;
    const scale = window.devicePixelRatio || 1;
    chart.width = width * scale;
    chart.height = height * scale;
    const ctx = chart.getContext('2d');
    ctx.scale(scale, scale);
    ctx.font = '12px system-ui';
    const values = points.map(([, value]) => value);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const first = points[0][0];
    const last = points[points.length - 1][0];
    const left = Math.max(ctx.measureText(stats.format(low)).width, ctx.measureText(stats.format(high)).width) + 10;
    const right = width - 12;
    const top = 12;
    const bottom = height - 23;
    const x = iteration => left + (iteration - first) * (right - left) / (last - first || 1);
    const y = value => high === low ? (top + bottom) / 2 : bottom - (value - low) * (bottom - top) / (high - low);

    ctx.fillStyle = '#aab9bc';
    ctx.strokeStyle = '#303d43';
    for (const value of new Set([low, high])) {
      const tick = y(value);
      ctx.fillText(stats.format(value), 0, tick + 4);
      ctx.beginPath();
      ctx.moveTo(left, tick);
      ctx.lineTo(right, tick);
      ctx.stroke();
    }
    ctx.fillText(String(first), left, height - 4);
    if (last !== first) {
      ctx.textAlign = 'right';
      ctx.fillText(String(last), right, height - 4);
    }

    ctx.strokeStyle = '#91dfbb';
    ctx.fillStyle = '#91dfbb';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x(first), y(values[0]));
    for (const [iteration, value] of points.slice(1)) ctx.lineTo(x(iteration), y(value));
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x(last), y(values[values.length - 1]), 3, 0, 2 * Math.PI);
    ctx.fill();
  }

  select.addEventListener('change', drawChart);
  new ResizeObserver(drawChart).observe(chart);
}
