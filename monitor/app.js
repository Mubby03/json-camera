/* Live view of a json-camera run. Polls /api/status, draws inline SVG.
   No libraries: one page, one fetch, one render pass.

   Adapted from the YOLO monitor. The chart machinery is unchanged; what moved
   is the direction of "good". A detector's metrics all improve upwards, so the
   original could colour every rise green. Half of these do not: bits per pixel
   and loss improve by falling, so each series declares which way is better and
   the deltas are coloured from that rather than from the sign. */

'use strict';

const POLL_MS = 2000;
const $ = (id) => document.getElementById(id);

function f2(v) { return v == null ? '—' : v.toFixed(2); }
function f3(v) { return v == null ? '—' : v.toFixed(3); }
function fdB(v) { return v == null ? '—' : v.toFixed(2) + ' dB'; }

/* Series identity is fixed here and nowhere else, so a colour always means the
   same measure across every chart, tile and tooltip. Validation takes slots 1
   and 3, training takes 2 and 4 — held-out numbers are the ones that count, so
   they get the strongest hues. `better` drives delta colouring. */
const SERIES = {
  val_psnr:   { label: 'Val PSNR',   color: 'var(--series-1)', fmt: fdB, better: 'up' },
  train_psnr: { label: 'Train PSNR', color: 'var(--series-2)', fmt: fdB, better: 'up' },
  val_bpp:    { label: 'Val bpp',    color: 'var(--series-3)', fmt: f3,  better: 'down' },
  train_bpp:  { label: 'Train bpp',  color: 'var(--series-4)', fmt: f3,  better: 'down' },
  val_loss:   { label: 'Val loss',   color: 'var(--series-1)', fmt: f3,  better: 'down' },
  train_loss: { label: 'Train loss', color: 'var(--series-2)', fmt: f3,  better: 'down' },
};

let latest = null;
let tableOpen = false;
// Charts are rebuilt only when there is a new epoch to draw. Redrawing on every
// 2s poll would tear down the SVG under the cursor and kill any open tooltip.
let drawnEpochs = -1;
let drawnTableEpochs = -1;

function clock(seconds) {
  if (seconds == null || !isFinite(seconds)) return '—';
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
  return `${s}s`;
}

function timeOfDay(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return isNaN(d) ? '—' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/* -- status bar ---------------------------------------------------------- */

function renderHead(s) {
  const pill = $('pill'), text = $('pill-text');
  pill.className = 'pill';
  if (s.error) { pill.classList.add('pill-err'); text.textContent = 'server error'; }
  else if (s.running) { pill.classList.add('pill-run'); text.textContent = 'training'; }
  else if ((s.epochs || []).length) { pill.classList.add('pill-done'); text.textContent = 'stopped'; }
  else { pill.classList.add('pill-idle'); text.textContent = 'waiting'; }

  $('stamp').textContent = new Date().toLocaleTimeString();

  const cfg = s.config || {};
  const bits = [];
  if (cfg.hidden && cfg.latent) bits.push(`hidden ${cfg.hidden} / latent ${cfg.latent}`);
  if (cfg.lmbda) bits.push(`lambda ${cfg.lmbda}`);
  if (cfg.train_patches) bits.push(`${Number(cfg.train_patches).toLocaleString()} patches`);
  if (cfg.device) bits.push(cfg.device);
  if (s.pid) bits.push(`pid ${s.pid}`);
  $('subtitle').textContent = s.error ? s.error : (s.message || bits.join(' · ') || 'no run found yet');
  if (cfg.val_patches != null) $('val-count').textContent = Number(cfg.val_patches).toLocaleString();
}

/* -- progress ------------------------------------------------------------ */

function renderProgress(s) {
  const eta = s.eta, cur = s.current, cfg = s.config || {};
  const total = Number(cfg.epochs_total || cfg.epochs || 0);
  const done = (s.epochs || []).length;

  $('epoch-done').textContent = done;
  $('epoch-total').textContent = total || '—';

  const frac = eta ? eta.fraction_complete : (total ? done / total : 0);
  const pct = Math.max(0, Math.min(1, frac || 0));
  $('meter-overall').style.width = (pct * 100).toFixed(2) + '%';
  $('overall-pct').textContent = (pct * 100).toFixed(1) + '%';
  $('meter-overall-wrap').setAttribute('aria-valuenow', (pct * 100).toFixed(0));

  if (eta) {
    $('finish').textContent = s.running ? timeOfDay(eta.finish) : '—';
    $('overall-time').textContent =
      `${clock(eta.elapsed_seconds)} elapsed · ${clock(eta.remaining_seconds)} left · ` +
      `${clock(eta.seconds_per_epoch)}/epoch (${eta.basis})`;
  } else {
    $('finish').textContent = '—';
    $('overall-time').textContent = 'no completed epoch to estimate from yet';
  }

  if (cur) {
    const p = cur.iterations ? cur.iteration / cur.iterations : 0;
    $('iter').textContent = cur.iteration.toLocaleString();
    $('iters').textContent = cur.iterations.toLocaleString();
    $('meter-epoch').style.width = (p * 100).toFixed(2) + '%';
    $('epoch-pct').textContent = (p * 100).toFixed(0) + '%';
    $('rate').textContent = fdB(cur.psnr);
    $('epoch-time').textContent =
      `epoch ${cur.epoch} · loss ${f3(cur.loss)} · ${f3(cur.bpp)} bpp`;
  } else {
    $('iter').textContent = '0'; $('iters').textContent = '0';
    $('meter-epoch').style.width = '0%';
    $('epoch-pct').textContent = '0%';
    $('rate').textContent = '—';
    $('epoch-time').textContent = s.running ? 'between epochs — validating' : 'idle';
  }
}

/* -- stat tiles ---------------------------------------------------------- */

function tile(host, { label, color, value, deltaText, deltaClass }) {
  const node = document.createElement('div');
  node.className = 'tile';

  const head = document.createElement('div');
  head.className = 'tile-label';
  if (color) {
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = color;
    head.append(swatch);
  }
  head.append(document.createTextNode(label));

  const val = document.createElement('div');
  val.className = 'tile-value';
  val.textContent = value;

  const delta = document.createElement('div');
  delta.className = 'tile-delta';
  delta.textContent = deltaText;
  if (deltaClass) delta.classList.add(deltaClass);

  node.append(head, val, delta);
  host.append(node);
}

function renderTiles(s) {
  const rows = s.epochs || [];
  const last = rows[rows.length - 1];
  const prev = rows[rows.length - 2];
  const host = $('tiles');
  host.innerHTML = '';

  for (const key of ['val_psnr', 'val_bpp', 'val_loss']) {
    const meta = SERIES[key];
    const value = last ? last[key] : null;
    const before = prev ? prev[key] : null;

    let deltaText = last ? `epoch ${last.epoch}` : 'no epoch finished yet';
    let deltaClass = '';
    if (value != null && before != null) {
      const d = value - before;
      // Direction of improvement is per-metric: bpp and loss get better going
      // down, so the sign alone cannot decide the colour.
      const improved = meta.better === 'up' ? d > 0 : d < 0;
      deltaText = `${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(3)} vs epoch ${prev.epoch}`;
      deltaClass = improved ? 'up' : 'down';
    }
    tile(host, { label: meta.label, color: meta.color, value: meta.fmt(value), deltaText, deltaClass });
  }

  // Compression ratio is the number a storage system actually cares about, and
  // it is just the rate restated: raw RGB is 24 bits per pixel.
  const bpp = last ? last.val_bpp : null;
  tile(host, {
    label: 'Compression vs raw',
    value: bpp ? `${(24 / bpp).toFixed(0)}×` : '—',
    deltaText: bpp ? `${f3(bpp)} bpp of 24` : 'waiting for epoch 1',
  });
}

/* -- charts -------------------------------------------------------------- */

const NS = 'http://www.w3.org/2000/svg';
const el = (name, attrs) => {
  const node = document.createElementNS(NS, name);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  return node;
};

function niceTicks(lo, hi, count) {
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const raw = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || mag * 10;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(t);
  return out;
}

function drawChart(host, rows, keys, opts) {
  host.innerHTML = '';
  opts = opts || {};

  const legend = document.createElement('ul');
  legend.className = 'legend';
  for (const key of keys) {
    const li = document.createElement('li');
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = SERIES[key].color;
    li.append(sw, document.createTextNode(SERIES[key].label));
    legend.append(li);
  }
  host.append(legend);

  const usable = rows.filter(r => keys.some(k => r[k] != null));
  if (usable.length === 0) {
    const note = document.createElement('div');
    note.className = 'empty';
    note.textContent = 'Nothing plotted yet — the first point appears when epoch 1 finishes validating.';
    host.append(note);
    return;
  }
  rows = usable;

  const width = Math.max(host.clientWidth || 640, 320);
  const height = opts.height || 260;
  const pad = { t: 10, r: 76, b: 30, l: 52 };
  const iw = width - pad.l - pad.r;
  const ih = height - pad.t - pad.b;

  const xs = rows.map(r => r.epoch);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const values = keys.flatMap(k => rows.map(r => r[k]).filter(v => v != null));
  let yMin = Math.min(...values), yMax = Math.max(...values);
  if (opts.zeroFloor) yMin = Math.min(0, yMin);
  const padY = (yMax - yMin) * 0.12 || 0.05;
  yMin -= padY; yMax += padY;

  const X = v => pad.l + (xMax === xMin ? iw / 2 : (v - xMin) / (xMax - xMin) * iw);
  const Y = v => pad.t + ih - (v - yMin) / (yMax - yMin) * ih;

  const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img' });
  svg.setAttribute('aria-label', opts.label || 'training metrics over epochs');

  for (const t of niceTicks(yMin, yMax, 4)) {
    const y = Y(t);
    if (y < pad.t - 1 || y > pad.t + ih + 1) continue;
    svg.append(el('line', { class: 'grid-line', x1: pad.l, x2: pad.l + iw, y1: y, y2: y }));
    const label = el('text', { class: 'tick-text', x: pad.l - 8, y: y + 3.5, 'text-anchor': 'end' });
    label.textContent = Math.abs(t) < 1 ? t.toFixed(2) : t.toFixed(1);
    svg.append(label);
  }

  svg.append(el('line', { class: 'axis-line', x1: pad.l, x2: pad.l + iw, y1: pad.t + ih, y2: pad.t + ih }));

  const step = Math.max(1, Math.ceil(rows.length / 8));
  for (let i = 0; i < rows.length; i += step) {
    const x = X(rows[i].epoch);
    const label = el('text', { class: 'tick-text', x, y: pad.t + ih + 16, 'text-anchor': 'middle' });
    label.textContent = rows[i].epoch;
    svg.append(label);
  }
  const axisTitle = el('text', { class: 'axis-title', x: pad.l + iw / 2, y: height - 1, 'text-anchor': 'middle' });
  axisTitle.textContent = 'epoch';
  svg.append(axisTitle);

  const ends = [];
  for (const key of keys) {
    const points = rows.filter(r => r[key] != null);
    if (!points.length) continue;
    const d = points.map((r, i) => `${i ? 'L' : 'M'}${X(r.epoch).toFixed(2)},${Y(r[key]).toFixed(2)}`).join(' ');
    svg.append(el('path', { class: 'series-line', d, stroke: SERIES[key].color }));

    /* One marker per point while the run is short enough to read; past that the
       line carries the shape and dots would only crowd it. */
    if (points.length <= 30) {
      for (const r of points) {
        svg.append(el('circle', {
          class: 'series-dot', cx: X(r.epoch), cy: Y(r[key]), r: 4, fill: SERIES[key].color,
        }));
      }
    }
    const last = points[points.length - 1];
    ends.push({ key, x: X(last.epoch), y: Y(last[key]) });
  }

  /* Direct labels — the relief the palette validator requires for the two
     light-mode hues below 3:1, and quicker to read than a legend hop anyway.
     Nudge any pair closer than 13px apart so they never overlap. */
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) {
    if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
  }
  for (const e of ends) {
    const label = el('text', { class: 'end-label', x: e.x + 9, y: Math.min(e.y + 3.5, height - 4) });
    label.setAttribute('fill', SERIES[e.key].color);
    label.textContent = SERIES[e.key].label;
    svg.append(label);
  }

  // Parked with opacity rather than off-canvas: the svg is overflow:visible so
  // a line at x=-99 is still drawn, in the page margin.
  const cross = el('line', { class: 'crosshair', y1: pad.t, y2: pad.t + ih, x1: 0, x2: 0, opacity: 0 });
  svg.append(cross);
  const hit = el('rect', { class: 'hit', x: pad.l, y: pad.t, width: iw, height: ih });
  svg.append(hit);
  host.append(svg);

  const tip = document.createElement('div');
  tip.className = 'tooltip';
  host.append(tip);

  const move = (event) => {
    const box = svg.getBoundingClientRect();
    const scale = width / box.width;
    const mx = (event.clientX - box.left) * scale;
    let best = rows[0], bestD = Infinity;
    for (const r of rows) {
      const d = Math.abs(X(r.epoch) - mx);
      if (d < bestD) { bestD = d; best = r; }
    }
    const x = X(best.epoch);
    cross.setAttribute('x1', x); cross.setAttribute('x2', x);
    cross.setAttribute('opacity', 1);

    let html = `<div class="tooltip-head">Epoch ${best.epoch}</div>`;
    for (const key of keys) {
      html += `<div class="tooltip-row"><span><span class="swatch" style="background:${SERIES[key].color}"></span>` +
              `${SERIES[key].label}</span><b>${SERIES[key].fmt(best[key])}</b></div>`;
    }
    tip.innerHTML = html;
    tip.classList.add('on');
    const left = x / scale;
    tip.style.left = Math.min(Math.max(left + 14, 4), box.width - tip.offsetWidth - 4) + 'px';
    tip.style.top = '8px';
  };
  hit.addEventListener('mousemove', move);
  hit.addEventListener('mouseleave', () => {
    tip.classList.remove('on');
    cross.setAttribute('opacity', 0);
  });
}

function renderCharts(s, force) {
  const rows = s.epochs || [];
  if (!force && rows.length === drawnEpochs) return;
  drawnEpochs = rows.length;
  drawChart($('chart-psnr'), rows, ['val_psnr', 'train_psnr'], { label: 'PSNR over epochs' });
  drawChart($('chart-bpp'), rows, ['val_bpp', 'train_bpp'], { zeroFloor: true, label: 'bits per pixel over epochs' });
  drawChart($('chart-loss'), rows, ['val_loss', 'train_loss'], { zeroFloor: true, label: 'rate-distortion loss over epochs' });
}

/* -- table, config, weights ---------------------------------------------- */

function renderTable(s, force) {
  const wrap = $('table-wrap');
  const rows = s.epochs || [];
  wrap.hidden = !tableOpen;
  $('table-toggle').textContent = tableOpen ? 'Hide table' : 'Show table';
  $('table-toggle').setAttribute('aria-expanded', String(tableOpen));
  if (!tableOpen) return;
  if (!force && rows.length === drawnTableEpochs) return;
  drawnTableEpochs = rows.length;

  if (!rows.length) { wrap.innerHTML = '<p class="empty">No completed epochs yet.</p>'; return; }
  const cols = [
    ['epoch', 'Epoch', 0], ['val_psnr', 'Val PSNR', 2], ['val_bpp', 'Val bpp', 4],
    ['val_loss', 'Val loss', 4], ['train_psnr', 'Train PSNR', 2],
    ['train_bpp', 'Train bpp', 4], ['train_loss', 'Train loss', 4],
    ['seconds', 'Seconds', 0],
  ];
  let html = '<div class="table-scroll"><table><thead><tr>';
  for (const [, label] of cols) html += `<th>${label}</th>`;
  html += '<th>Best</th></tr></thead><tbody>';
  for (const r of [...rows].reverse()) {
    html += '<tr>';
    for (const [key, , dp] of cols) {
      const v = r[key];
      html += `<td>${v == null ? '—' : v.toFixed(dp)}</td>`;
    }
    html += `<td>${r.best ? '✓' : ''}</td></tr>`;
  }
  wrap.innerHTML = html + '</tbody></table></div>';
}

function renderConfig(s) {
  const cfg = s.config || {};
  const pairs = [
    ['lambda (quality knob)', cfg.lmbda], ['hidden', cfg.hidden], ['latent', cfg.latent],
    ['epochs', cfg.epochs_total || cfg.epochs], ['batch', cfg.batch],
    ['learning rate', cfg.lr], ['device', cfg.device], ['workers', cfg.workers],
    ['steps per epoch', cfg.steps_per_epoch ? Number(cfg.steps_per_epoch).toLocaleString() : null],
    ['train patches', cfg.train_patches ? Number(cfg.train_patches).toLocaleString() : null],
    ['held-out patches', cfg.val_patches ? Number(cfg.val_patches).toLocaleString() : null],
    ['best epoch so far', s.best],
    ['config source', cfg._inferred ? 'INFERRED: ' + cfg._inferred : 'the run log itself'],
    ['log', s.log],
  ].filter(([, v]) => v != null && v !== '');
  $('config').innerHTML = pairs
    .map(([k, v]) => `<dt>${k}</dt><dd>${String(v).replace(/^.*\/(?=[^/]*\/[^/]*$)/, '…/')}</dd>`)
    .join('');
}

function renderWeights(s) {
  const list = s.weights || [];
  if (!list.length) {
    $('weights').innerHTML = '<dt>none yet</dt><dd>written when epoch 1 finishes</dd>';
    return;
  }
  $('weights').innerHTML = list.map(w =>
    `<dt>${w.name}</dt><dd>${(w.bytes / 1048576).toFixed(1)} MB · ${timeOfDay(w.modified)}</dd>`).join('');
}

/* -- poll ---------------------------------------------------------------- */

function render(s) {
  latest = s;
  renderHead(s);
  renderProgress(s);
  renderTiles(s);
  renderCharts(s);
  renderTable(s);
  renderConfig(s);
  renderWeights(s);
}

async function poll() {
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    render(await response.json());
  } catch (error) {
    // The server being down is a state to display, not an exception to swallow.
    render({ error: 'cannot reach the monitor server — is server.py still running?',
             running: false, epochs: [], weights: [] });
  }
}

$('table-toggle').addEventListener('click', () => {
  tableOpen = !tableOpen;
  if (latest) renderTable(latest, true);
});

let resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (latest) renderCharts(latest, true); }, 150);
});

poll();
setInterval(poll, POLL_MS);
