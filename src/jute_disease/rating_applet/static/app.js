const $ = id => document.getElementById(id);
let state, item, position, busy = false, readable = false, failed = false, history = [], full = false;
const groups = {
  mask_quality: { title: 'Mask quality', options: [['pass', 'Pass', '1'], ['minor', 'Minor', '2'], ['fail', 'Fail', '3']] },
  scene_tag: { title: 'Scene', options: [['clean-background', 'Clean background', 'Q'], ['field', 'Field', 'W'], ['unknown', 'Unknown', 'E']] }
};
async function json(url, options) { const r = await fetch(url, options); const data = await r.json(); if (!r.ok) throw Error(data.error || 'Request failed.'); return data; }
async function refresh(expected) { const latest = await json('/api/sitting'); if (expected && latest.revision !== expected) throw Error('The sitting changed. Reload it.'); state = latest; $('progress').textContent = `${state.completed} completed · ${state.total - state.completed} remaining`; $('mode').textContent = state.read_only ? 'Read-only' : ''; }
function complete() { return item && item.scene_tag && (item.kind !== 'mask' || item.mask_quality); }
function controls() {
  document.querySelectorAll('.choice button').forEach(button => { button.disabled = busy || failed || state.read_only; button.setAttribute('aria-pressed', String(item[button.dataset.field] === button.dataset.value)); });
  $('previous').disabled = busy || failed || position <= 1;
  $('next').disabled = busy || failed || !readable || !complete() || position >= state.total;
  $('finish').disabled = busy || failed || !readable || state.read_only || state.completed !== state.total;
  $('undo').disabled = busy || failed || state.read_only || !history.length;
}
function choices() {
  const container = $('choices'); container.replaceChildren();
  for (const field of item.kind === 'mask' ? ['mask_quality', 'scene_tag'] : ['scene_tag']) {
    const group = groups[field], set = document.createElement('fieldset'), legend = document.createElement('legend');
    legend.textContent = group.title; set.append(legend);
    for (const [value, label, key] of group.options) {
      const row = document.createElement('div'), button = document.createElement('button'), anchor = document.createElement('p');
      row.className = 'choice'; button.type = 'button'; button.textContent = label; button.dataset.field = field; button.dataset.value = value;
      button.setAttribute('aria-keyshortcuts', key); button.title = `Shortcut: ${key}`;
      button.addEventListener('click', () => save(field, value));
      const shortcut = document.createElement('span'); shortcut.textContent = ` (${key})`; shortcut.setAttribute('aria-hidden', 'true'); button.append(shortcut);
      anchor.textContent = (field === 'mask_quality' ? item.mask_anchors : item.scene_anchors)[value] || '';
      row.append(button, anchor); set.append(row);
    }
    container.append(set);
  }
  controls();
}
async function show(pos) {
  position = pos; history = []; full = false; readable = false; $('save').textContent = ''; $('image-status').textContent = 'Loading image…';
  $('position').textContent = `Item ${position} of ${state.total}`;
  try { item = await json(`/api/items/${position}`); choices(); } catch (e) { item = null; $('choices').replaceChildren(); $('notice').textContent = e.message; return; }
  const picture = $('picture'); picture.className = ''; picture.style.width = ''; $('scale').value = 1; $('zoom').textContent = 'Inspect full resolution';
  picture.onload = () => { readable = true; $('image-status').textContent = ''; controls(); };
  picture.onerror = () => { readable = false; $('image-status').textContent = 'Image unavailable. Resolve the issue before advancing.'; controls(); };
  picture.src = `/images/${position}/preview`; controls();
}
async function save(field, value, undo = false) {
  if (busy || failed || state.read_only || value === item[field]) return;
  const old = item[field]; busy = true; $('save').textContent = 'Saving…'; controls();
  try {
    const result = await json(`/api/items/${position}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({revision: state.revision, field, value})});
    state.revision = result.revision; item[field] = value; if (!undo) history.push({field, value: old});
    await refresh(); $('save').textContent = 'Saved';
  } catch (e) { failed = true; $('save').textContent = `${e.message} Reload to verify the saved state.`; }
  busy = false; controls();
}
$('previous').onclick = () => show(position - 1); $('next').onclick = () => show(position + 1);
$('undo').onclick = async () => { const change = history.pop(); if (change) await save(change.field, change.value, true); };
$('finish').onclick = async () => { busy = true; controls(); try { await json('/api/finish', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({revision: state.revision})}); await refresh(); $('notice').textContent = 'Sitting finished. Take a break before the next sitting.'; } catch (e) { failed = true; $('notice').textContent = e.message; } busy = false; controls(); };
$('zoom').onclick = () => { full = !full; readable = false; $('image-status').textContent = 'Loading image…'; $('picture').classList.toggle('full', full); $('picture').style.width = ''; $('scale').value = 1; $('zoom').textContent = full ? 'Fit preview' : 'Inspect full resolution'; $('picture').src = `/images/${position}/${full ? 'original' : 'preview'}`; controls(); };
$('scale').oninput = () => { if (full) $('picture').style.width = `${$('picture').naturalWidth * Number($('scale').value)}px`; };
document.addEventListener('keydown', e => { if (e.ctrlKey || e.altKey || e.metaKey || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable || e.target.id === "zoom" || !item) return; for (const [field, group] of Object.entries(groups)) { if (field === 'mask_quality' && item.kind !== 'mask') continue; const choice = group.options.find(option => option[2].toLowerCase() === e.key.toLowerCase()); if (choice) { e.preventDefault(); save(field, choice[0]); return; } } });
(async () => { try { await refresh(); await show(state.position); } catch (e) { $('notice').textContent = e.message; } })();


