/* Shared front end helpers. No libraries and no build step. */

'use strict';

const JC = (function () {

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function bytes(n) {
    if (n == null) return '-';
    n = Number(n);
    const units = ['B', 'KB', 'MB', 'GB'];
    for (let i = 0; i < units.length; i++) {
      if (n < 1024 || i === units.length - 1) {
        return i === 0 ? Math.round(n) + ' B' : n.toFixed(1) + ' ' + units[i];
      }
      n /= 1024;
    }
  }

  /* A dropzone that is also a real label wrapping a real file input, so the
     keyboard and screen reader paths work without any extra handling. */
  function dropzone(zone, input, onPick) {
    const stop = (e) => { e.preventDefault(); e.stopPropagation(); };

    ['dragenter', 'dragover'].forEach(ev =>
      zone.addEventListener(ev, (e) => { stop(e); zone.classList.add('hot'); }));
    ['dragleave', 'drop'].forEach(ev =>
      zone.addEventListener(ev, (e) => { stop(e); zone.classList.remove('hot'); }));

    zone.addEventListener('drop', (e) => {
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (!file) return;
      // Put it on the input too, so the form still reflects what is staged.
      try {
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
      } catch (_) { /* older browsers: the callback still has the file */ }
      onPick(file);
    });

    input.addEventListener('change', () => onPick(input.files[0] || null));

    // The whole page is a drop target for the sake of near misses, but only the
    // zone lights up, and a drop outside it must not navigate away from the app.
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => e.preventDefault());
  }

  function showPicked(f) {
    const box = document.getElementById('picked');
    if (!f) { box.hidden = true; return; }
    document.getElementById('picked-name').textContent = f.name;
    document.getElementById('picked-size').textContent = bytes(f.size);
    box.hidden = false;
  }

  async function fillModels(select, hint, allowAuto) {
    try {
      const r = await fetch('/api/models');
      const data = await r.json();
      select.innerHTML = '';

      if (allowAuto) {
        const auto = document.createElement('option');
        auto.value = '';
        auto.textContent = 'Automatic, match the file';
        select.append(auto);
      }

      if (!data.models.length) {
        const none = document.createElement('option');
        none.textContent = 'No trained model available';
        select.append(none);
        select.disabled = true;
        return;
      }

      for (const m of data.models) {
        const o = document.createElement('option');
        o.value = m.id;
        const bits = [];
        if (m.lmbda) bits.push('lambda ' + m.lmbda);
        // Held out figures, which is what the server sends when it has them.
        if (m.bpp) bits.push(m.bpp.toFixed(2) + ' bpp');
        if (m.psnr) bits.push(m.psnr.toFixed(1) + ' dB');
        o.textContent = m.label + (bits.length ? '  (' + bits.join(', ') + ')' : '');
        select.append(o);
      }
      if (hint && data.max_side) {
        hint.textContent = data.models.length > 1
          ? 'One model per quality level, which is what lambda buys. The small one beats '
            + 'JPEG at a matched file size; the sharp one looks better but no longer does, '
            + 'because JPEG is strong at higher bitrates. Images over '
            + data.max_side.toLocaleString() + ' pixels are scaled down first.'
          : 'Lambda is the quality knob. Higher means a better picture and a bigger file. '
            + 'Images wider or taller than ' + data.max_side.toLocaleString() + ' pixels are scaled down first.';
      }
    } catch (_) {
      select.innerHTML = '<option>Could not reach the server</option>';
      select.disabled = true;
    }
  }

  function busy(out, message) {
    out.innerHTML =
      '<div class="bar"><i></i></div>' +
      '<div class="busy" style="margin-top:0"><span class="spin"></span><span>' + esc(message) + '</span></div>';
  }

  function fail(out, message) {
    out.innerHTML = '<div class="alert"><b>That did not work.</b>' + esc(message) + '</div>';
  }

  async function post(url, body) {
    const r = await fetch(url, { method: 'POST', body });
    if (!r.ok) {
      let detail = 'The server returned ' + r.status + '.';
      try {
        const j = await r.json();
        if (j && j.detail) detail = j.detail;
      } catch (_) { /* not json, keep the status line */ }
      throw new Error(detail);
    }
    return r.json();
  }

  return { esc, bytes, dropzone, showPicked, fillModels, busy, fail, post };
})();
