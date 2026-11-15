const $ = id => document.getElementById(id);
let state;
let item;
let position;
let busy = false;
let readable = false;
let failed = false;
let history = [];
let full = false;
let actualPixels = false;
let swapped = false;
let imageGeneration = 0;
let finishArmed = false;
let syncingPan = false;

const groups = {
  rating: {
    title: 'Pair rating',
    options: [
      ['exact', 'Exact', '1'],
      ['near-duplicate', 'Near-duplicate', '2'],
      ['very similar', 'Very similar', '3'],
      ['different', 'Different', '4']
    ]
  },
  mask_quality: {
    title: 'Mask quality',
    options: [['pass', 'Pass', '1'], ['minor', 'Minor', '2'], ['fail', 'Fail', '3']]
  },
  scene_tag: {
    title: 'Scene',
    options: [['clean-background', 'Clean background', 'Q'], ['field', 'Field', 'W'], ['unknown', 'Unknown', 'E']]
  }
};

async function json(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw Error(data.error || 'Request failed.');
  return data;
}

async function refresh(expected) {
  const latest = await json('/api/sitting');
  if (expected && latest.revision !== expected) throw Error('The sitting changed. Reload it.');
  state = latest;
  $('progress').textContent = `${state.completed} completed · ${state.total - state.completed} remaining`;
  $('mode').textContent = state.read_only ? 'Read-only' : '';
}

function complete() {
  if (!item) return false;
  if (item.kind === 'pair') return Boolean(item.rating);
  return Boolean(item.scene_tag && (item.kind !== 'mask' || item.mask_quality));
}

function controls() {
  document.querySelectorAll('.choice button').forEach(button => {
    button.disabled = busy || failed || state.read_only || !readable;
    button.setAttribute('aria-pressed', String(item[button.dataset.field] === button.dataset.value));
  });
  document.querySelectorAll('.zoom button, .zoom input').forEach(control => {
    control.disabled = busy;
  });
  $('previous').disabled = busy || failed || position <= 1;
  $('next').disabled = busy || failed || !readable || !complete() || position >= state.total;
  $('finish').disabled = busy || failed || !readable || state.read_only || state.completed !== state.total;
  $('cancel-finish').disabled = busy;
  $('undo').disabled = busy || failed || state.read_only || !history.length;
}

function addPairRubric(container) {
  const rubric = document.createElement('section');
  rubric.className = 'pair-rubric';
  rubric.setAttribute('aria-label', 'Pair rating definitions');
  const heading = document.createElement('h3');
  heading.textContent = 'Pair rating definitions';
  rubric.append(heading);
  const near = document.createElement('p');
  near.textContent = `Near-duplicate: “${item.pair_boundary.near_duplicate}.”`;
  const similar = document.createElement('p');
  similar.textContent = `Very similar: “${item.pair_boundary.very_similar}.”`;
  const merge = document.createElement('p');
  merge.textContent = item.pair_boundary.merge;
  rubric.append(near, similar, merge);
  container.append(rubric);
}

function choices() {
  const container = $('choices');
  container.replaceChildren();
  const fields = item.kind === 'pair'
    ? ['rating']
    : item.kind === 'mask' ? ['mask_quality', 'scene_tag'] : ['scene_tag'];
  for (const field of fields) {
    const group = groups[field];
    const set = document.createElement('fieldset');
    const legend = document.createElement('legend');
    legend.textContent = group.title;
    set.append(legend);
    for (const [value, label, key] of group.options) {
      const row = document.createElement('div');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'rating-choice';
      button.textContent = label;
      button.dataset.field = field;
      button.dataset.value = value;
      button.setAttribute('aria-keyshortcuts', key);
      button.title = `Shortcut: ${key}`;
      button.addEventListener('click', () => save(field, value));
      const shortcut = document.createElement('span');
      shortcut.className = 'shortcut';
      shortcut.textContent = `(${key})`;
      button.append(shortcut);
      row.className = 'choice';
      row.append(button);
      if (field !== 'rating') {
        const anchor = document.createElement('p');
        anchor.textContent = (field === 'mask_quality' ? item.mask_anchors : item.scene_anchors)[value] || '';
        row.append(anchor);
      }
      set.append(row);
    }
    container.append(set);
  }
  if (item.kind === 'pair') addPairRubric(container);
  controls();
}

function makeImage(side) {
  const picture = document.createElement('img');
  picture.alt = side ? `Image ${side.toUpperCase()} for this item` : 'Image for this item';
  picture.dataset.side = side || '';
  return picture;
}

function imageUrl(side, variant) {
  return side
    ? `/images/${position}/${side}/${variant}`
    : `/images/${position}/${variant}`;
}

function applyScale() {
  const scale = actualPixels ? 1 : Number($('scale').value);
  document.querySelectorAll('#image-frame img, #pair-frames img').forEach(picture => {
    picture.style.width = full ? `${picture.naturalWidth * scale}px` : '';
  });
}

function syncPairPan(source) {
  if (syncingPan) return;
  const frames = [...document.querySelectorAll('.pair-image-frame')];
  const target = frames.find(frame => frame !== source);
  if (!target) return;
  syncingPan = true;
  const horizontalRange = source.scrollWidth - source.clientWidth;
  const verticalRange = source.scrollHeight - source.clientHeight;
  target.scrollLeft = horizontalRange > 0
    ? source.scrollLeft / horizontalRange * (target.scrollWidth - target.clientWidth)
    : 0;
  target.scrollTop = verticalRange > 0
    ? source.scrollTop / verticalRange * (target.scrollHeight - target.clientHeight)
    : 0;
  syncingPan = false;
}

function loadImages() {
  const generation = ++imageGeneration;
  const sides = item.kind === 'pair' ? ['a', 'b'] : [null];
  const pictures = [...document.querySelectorAll('#image-frame img, #pair-frames img')];
  let loaded = 0;
  readable = false;
  $('image-status').textContent = 'Loading image…';
  for (const picture of pictures) {
    const side = picture.dataset.side || null;
    picture.classList.toggle('full', full);
    picture.onload = () => {
      if (generation !== imageGeneration) return;
      loaded += 1;
      if (loaded === sides.length) {
        readable = true;
        $('image-status').textContent = '';
        applyScale();
        controls();
      }
    };
    picture.onerror = () => {
      if (generation !== imageGeneration) return;
      readable = false;
      $('image-status').textContent = 'Image unavailable. Resolve the issue before advancing.';
      controls();
    };
    picture.src = imageUrl(side, full ? 'original' : 'preview');
  }
  controls();
}

function setImageSources() {
  const sides = item.kind === 'pair' ? ['a', 'b'] : [null];
  const pictures = sides.map(side => makeImage(side));
  $('image-frame').hidden = item.kind === 'pair';
  $('image-frame').replaceChildren();
  $('pair-frames').hidden = item.kind !== 'pair';
  $('pair-frames').replaceChildren();
  if (item.kind === 'pair') {
    for (const [index, side] of sides.entries()) {
      const figure = document.createElement('figure');
      const caption = document.createElement('figcaption');
      const frame = document.createElement('div');
      figure.className = 'pair-panel';
      figure.dataset.side = side;
      caption.textContent = `Image ${side.toUpperCase()}`;
      frame.className = 'image-frame pair-image-frame';
      frame.append(pictures[index]);
      frame.addEventListener('scroll', () => syncPairPan(frame), { passive: true });
      figure.append(caption, frame);
      $('pair-frames').append(figure);
    }
    $('swap').hidden = false;
    $('actual-pixels').hidden = false;
    $('actual-pixels-note').hidden = false;
  } else {
    $('image-frame').append(pictures[0]);
    $('swap').hidden = true;
    $('actual-pixels').hidden = true;
    $('actual-pixels-note').hidden = true;
  }
  loadImages();
}

async function show(nextPosition) {
  if (busy || failed || !state) return;
  busy = true;
  controls();
  try {
    const nextItem = await json(`/api/items/${nextPosition}`);
    position = nextPosition;
    item = nextItem;
    history = [];
    full = false;
    actualPixels = false;
    swapped = false;
    finishArmed = false;
    readable = false;
    failed = false;
    $('save').textContent = '';
    $('image-status').textContent = 'Loading image…';
    $('position').textContent = `Item ${position} of ${state.total}`;
    $('scale').value = 1;
    $('zoom').textContent = 'Inspect full resolution';
    $('actual-pixels').textContent = '1:1 image pixels';
    $('actual-pixels').title = 'One image pixel per CSS pixel; monitor scaling may change physical size.';
    $('actual-pixels').setAttribute('aria-pressed', 'false');
    $('swap').setAttribute('aria-pressed', 'false');
    $('finish').textContent = 'Finish sitting';
    $('cancel-finish').hidden = true;
    $('notice').textContent = '';
    choices();
    setImageSources();
  } catch (error) {
    $('notice').textContent = error.message;
  }
  busy = false;
  controls();
}

async function save(field, value, undo = false) {
  if (busy || failed || !readable || state.read_only || value === item[field]) return;
  const oldValue = item[field];
  busy = true;
  $('save').textContent = 'Saving…';
  controls();
  try {
    const result = await json(`/api/items/${position}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ revision: state.revision, field, value })
    });
    state.revision = result.revision;
    item[field] = value;
    if (!undo) history.push({ field, value: oldValue });
    await refresh(result.revision);
    sessionStorage.removeItem('rating-applet-recovery-position');
    $('save').textContent = 'Saved';
    $('notice').textContent = '';
  } catch (error) {
    failed = true;
    sessionStorage.setItem('rating-applet-recovery-position', String(position));
    $('save').textContent = `${error.message} Reload to verify the saved state.`;
  }
  busy = false;
  controls();
}

$('previous').onclick = () => show(position - 1);
$('next').onclick = () => show(position + 1);
$('undo').onclick = async () => {
  const change = history.pop();
  if (change) await save(change.field, change.value, true);
};
$('finish').onclick = async () => {
  if (!finishArmed) {
    finishArmed = true;
    $('finish').textContent = 'Confirm finish';
    $('cancel-finish').hidden = false;
    $('notice').textContent = 'Finishing makes this sitting read-only. Confirm to finish.';
    return;
  }
  busy = true;
  controls();
  try {
    await json('/api/finish', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ revision: state.revision })
    });
    await refresh();
    finishArmed = false;
    $('finish').textContent = 'Finish sitting';
    $('cancel-finish').hidden = true;
    $('notice').textContent = 'Sitting finished. Take a break before the next sitting.';
  } catch (error) {
    failed = true;
    $('notice').textContent = error.message;
  }
  busy = false;
  controls();
};
$('cancel-finish').onclick = () => {
  finishArmed = false;
  $('finish').textContent = 'Finish sitting';
  $('cancel-finish').hidden = true;
  $('notice').textContent = '';
};
$('zoom').onclick = () => {
  full = !full;
  actualPixels = false;
  $('scale').value = 1;
  $('actual-pixels').setAttribute('aria-pressed', 'false');
  $('zoom').textContent = full ? 'Fit preview' : 'Inspect full resolution';
  $('image-status').textContent = 'Loading image…';
  loadImages();
};
$('actual-pixels').onclick = () => {
  if (!full) {
    full = true;
    $('zoom').textContent = 'Fit preview';
    $('scale').value = 1;
    actualPixels = true;
    $('actual-pixels').setAttribute('aria-pressed', 'true');
    $('actual-pixels').textContent = 'Use zoom scale';
    loadImages();
    return;
  }
  actualPixels = !actualPixels;
  $('actual-pixels').setAttribute('aria-pressed', String(actualPixels));
  $('actual-pixels').textContent = actualPixels ? 'Use zoom scale' : '1:1 image pixels';
  if (actualPixels) $('scale').value = 1;
  applyScale();
};
$('swap').onclick = () => {
  swapped = !swapped;
  document.querySelectorAll('.pair-panel').forEach(panel => {
    panel.style.order = (panel.dataset.side === 'a') === swapped ? '2' : '1';
  });
  $('swap').setAttribute('aria-pressed', String(swapped));
};
$('scale').oninput = () => {
  actualPixels = false;
  $('actual-pixels').setAttribute('aria-pressed', 'false');
  $('actual-pixels').textContent = '1:1 image pixels';
  applyScale();
};
document.addEventListener('keydown', event => {
  const target = event.target;
  if (
    busy || failed || event.ctrlKey || event.altKey || event.metaKey ||
    !item
  ) return;
  if (event.key === '[') {
    event.preventDefault();
    if (position > 1) show(position - 1);
    return;
  }
  if (event.key === ']') {
    event.preventDefault();
    if (readable && complete() && position < state.total) show(position + 1);
    return;
  }
  if (
    /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) ||
    target.isContentEditable
  ) return;
  if (!readable) return;
  const fields = item.kind === 'pair'
    ? ['rating']
    : item.kind === 'mask' ? ['mask_quality', 'scene_tag'] : ['scene_tag'];
  for (const field of fields) {
    const choice = groups[field].options.find(option => option[2].toLowerCase() === event.key.toLowerCase());
    if (choice) {
      event.preventDefault();
      save(field, choice[0]);
      return;
    }
  }
});

(async () => {
  try {
    await refresh();
    const recoveryPosition = Number(
      sessionStorage.getItem('rating-applet-recovery-position')
    );
    const startPosition = Number.isInteger(recoveryPosition) &&
      recoveryPosition >= 1 && recoveryPosition <= state.total
      ? recoveryPosition
      : state.position;
    await show(startPosition);
    if (startPosition === recoveryPosition) {
      $('notice').textContent = 'A previous save did not confirm. Review this item before continuing.';
    }
  } catch (error) {
    $('notice').textContent = error.message;
  }
})();
