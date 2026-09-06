"use strict";

/**
 * The 9 views, as (horizontal, vertical) camera-offset signs, matching
 * sharp_spatialize.cameras.DEFAULT_LAYOUT. Note the OpenCV convention: +Y is
 * down, so a +1 vertical sign puts that camera BELOW the reference view.
 */
const LAYOUT = [
  [-1, +1], [0, +1], [+1, +1],
  [-1, 0], [0, 0], [+1, 0],
  [-1, -1], [0, -1], [+1, -1],
];

/** View indices in the order a person sees them: top-left to bottom-right. */
const PHYSICAL_ORDER = [6, 7, 8, 3, 4, 5, 0, 1, 2];

/** A clockwise loop around the 8 outer cameras, for auto-orbit. */
const ORBIT_RING = [6, 7, 8, 5, 2, 1, 0, 3];

const REFERENCE_VIEW = 4;
const ORBIT_INTERVAL_MS = 260;
const POLL_INTERVAL_MS = 400;

const STAGE_TEXT = {
  queued: "Waiting for the GPU…",
  inference: "Running SHARP — one forward pass to 3D Gaussians…",
  rendering: "Rendering 9 views with gsplat…",
  validating: "Validating geometry and depth…",
  encoding: "Encoding previews…",
  done: "Done",
};

const el = (id) => document.getElementById(id);

const dom = {
  dropzone: el("dropzone"),
  fileInput: el("fileInput"),
  sampleBtn: el("sampleBtn"),
  angle: el("angle"),
  angleOut: el("angleOut"),
  maxSize: el("maxSize"),
  precision: el("precision"),
  runBtn: el("runBtn"),
  statusLine: el("statusLine"),
  deviceChip: el("deviceChip"),
  stage: el("stage"),
  layers: el("layers"),
  emptyState: el("emptyState"),
  progressOverlay: el("progressOverlay"),
  progressStage: el("progressStage"),
  progressBar: el("progressBar"),
  progressNote: el("progressNote"),
  errorOverlay: el("errorOverlay"),
  errorText: el("errorText"),
  indicator: el("indicator"),
  parallaxHint: el("parallaxHint"),
  sheet: el("sheet"),
  viewLabel: el("viewLabel"),
  orbitBtn: el("orbitBtn"),
  pinBtn: el("pinBtn"),
  infoCard: el("infoCard"),
  infoList: el("infoList"),
  validBadge: el("validBadge"),
  timingText: el("timingText"),
  downloadLink: el("downloadLink"),
};

const state = {
  config: null,
  file: null,
  useSample: false,
  jobId: null,
  running: false,
  ready: false,
  mode: "rgb",
  index: REFERENCE_VIEW,
  pinned: false,
  orbit: false,
  orbitTimer: null,
  orbitStep: 0,
  images: { rgb: [], depth: [] },
  angleDeg: 10,
  lastRunKey: null,
};

/* ---------------------------------------------------------------- helpers */

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

function viewLabel(index, angleDeg) {
  const [h, v] = LAYOUT[index];
  if (h === 0 && v === 0) return `V${index} · reference view`;
  const angle = Number(angleDeg).toFixed(angleDeg % 1 ? 1 : 0);
  const parts = [];
  if (v !== 0) parts.push(`${v < 0 ? "up" : "down"} ${angle}°`);
  if (h !== 0) parts.push(`${h < 0 ? "left" : "right"} ${angle}°`);
  return `V${index} · ${parts.join(" · ")}`;
}

function settingsKey() {
  return [
    state.useSample ? "sample" : state.file && state.file.name + state.file.size,
    dom.angle.value,
    dom.maxSize.value,
    dom.precision.value,
  ].join("|");
}

function refreshRunButton() {
  const hasInput = Boolean(state.file) || state.useSample;
  dom.runBtn.disabled = !hasInput || state.running;
  const changed = hasInput && state.lastRunKey !== null && state.lastRunKey !== settingsKey();
  dom.runBtn.classList.toggle("dirty", changed);
  if (!state.running) {
    dom.runBtn.textContent = changed ? "Regenerate" : "Generate spatial photo";
  }
}

function setStatus(message, isError = false) {
  dom.statusLine.textContent = message || "";
  dom.statusLine.classList.toggle("error", isError);
}

function updateSliderFill() {
  const input = dom.angle;
  const fraction = (input.value - input.min) / (input.max - input.min);
  input.style.setProperty("--fill", `${fraction * 100}%`);
  dom.angleOut.textContent = `${Number(input.value)}°`;
}

/* ------------------------------------------------------------- view state */

function showView(index) {
  state.index = index;
  for (const mode of ["rgb", "depth"]) {
    state.images[mode].forEach((img, i) => img.classList.toggle("active", i === index));
  }
  dom.viewLabel.textContent = viewLabel(index, state.angleDeg);
  Array.from(dom.indicator.children).forEach((cell, position) => {
    cell.classList.toggle("on", PHYSICAL_ORDER[position] === index);
  });
  Array.from(dom.sheet.children).forEach((thumb) => {
    thumb.classList.toggle("active", Number(thumb.dataset.index) === index);
  });
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".seg").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  state.images.rgb.forEach((img) => (img.hidden = mode !== "rgb"));
  state.images.depth.forEach((img) => (img.hidden = mode !== "depth"));
}

function indexFromPointer(event) {
  const bounds = dom.stage.getBoundingClientRect();
  const fx = clamp((event.clientX - bounds.left) / bounds.width, 0, 0.9999);
  const fy = clamp((event.clientY - bounds.top) / bounds.height, 0, 0.9999);
  const column = Math.floor(fx * 3);
  // Pointer at the top of the stage means "look from above", which is the
  // bottom row of LAYOUT (negative vertical sign, since +Y points down).
  const layoutRow = 2 - Math.floor(fy * 3);
  return layoutRow * 3 + column;
}

function setOrbit(enabled) {
  state.orbit = enabled;
  dom.orbitBtn.classList.toggle("active", enabled);
  clearInterval(state.orbitTimer);
  if (!enabled) return;
  state.orbitTimer = setInterval(() => {
    state.orbitStep = (state.orbitStep + 1) % ORBIT_RING.length;
    showView(ORBIT_RING[state.orbitStep]);
  }, ORBIT_INTERVAL_MS);
}

function setPinned(pinned) {
  state.pinned = pinned;
  dom.pinBtn.classList.toggle("active", pinned);
  dom.pinBtn.textContent = pinned ? "Unpin view" : "Pin view";
  dom.stage.classList.toggle("pinned", pinned);
}

/* ------------------------------------------------------------ the request */

async function loadConfig() {
  try {
    const config = await (await fetch("/api/config")).json();
    state.config = config;

    const device = config.device;
    dom.deviceChip.textContent = device.cuda
      ? `${device.name} · ${device.vram_gb} GB`
      : device.name;
    dom.deviceChip.classList.add(device.cuda ? "ok" : "warn");
    if (!device.cuda) {
      setStatus("Rendering needs a CUDA GPU; generation will fail on this machine.", true);
    }

    dom.maxSize.innerHTML = "";
    for (const size of config.max_size_choices) {
      const option = document.createElement("option");
      option.value = size;
      option.textContent = `${size} px long edge`;
      option.selected = size === config.defaults.max_size;
      dom.maxSize.append(option);
    }
    dom.angle.max = config.max_angle_deg;
    dom.angle.value = config.defaults.angle_deg;
    dom.precision.value = config.defaults.precision;
    updateSliderFill();

    dom.sampleBtn.hidden = !config.sample_available;
  } catch (error) {
    dom.deviceChip.textContent = "server unreachable";
    dom.deviceChip.classList.add("warn");
  }
}

async function startJob() {
  if (state.running || (!state.file && !state.useSample)) return;

  state.running = true;
  state.ready = false;
  state.lastRunKey = settingsKey();
  state.angleDeg = Number(dom.angle.value);
  setOrbit(false);
  setPinned(false);
  refreshRunButton();
  dom.runBtn.textContent = "Generating…";
  setStatus("");
  dom.emptyState.hidden = true;
  dom.errorOverlay.hidden = true;
  dom.progressOverlay.hidden = false;
  dom.indicator.hidden = true;
  dom.parallaxHint.hidden = true;
  dom.sheet.hidden = true;
  dom.infoCard.hidden = true;
  dom.layers.innerHTML = "";
  state.images = { rgb: [], depth: [] };
  showProgress("queued", 0.02);

  const params = new URLSearchParams({
    angle: dom.angle.value,
    max_size: dom.maxSize.value,
    precision: dom.precision.value,
  });
  let body;
  if (state.useSample) {
    params.set("sample", "1");
    body = new Blob([new Uint8Array(1)]);
  } else {
    params.set("filename", state.file.name);
    body = state.file;
  }

  try {
    const response = await fetch(`/api/jobs?${params}`, { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    state.jobId = payload.job_id;
    pollJob();
  } catch (error) {
    finishWithError(String(error.message || error));
  }
}

function showProgress(stage, progress, note = "") {
  dom.progressStage.textContent = STAGE_TEXT[stage] || stage;
  dom.progressBar.style.width = `${clamp(progress, 0.02, 1) * 100}%`;
  dom.progressNote.textContent = note;
}

async function pollJob() {
  const jobId = state.jobId;
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    const status = await response.json();
    if (!response.ok) throw new Error(status.error || `HTTP ${response.status}`);
    if (state.jobId !== jobId) return; // superseded by a newer run

    if (status.state === "error") return finishWithError(status.error);
    if (status.state === "done") return loadResults(status);

    const note = status.source && status.source.width
      ? `${status.source.width}×${status.source.height} · ${status.precision}`
      : "";
    showProgress(status.stage, status.progress, note);
    setTimeout(pollJob, POLL_INTERVAL_MS);
  } catch (error) {
    finishWithError(String(error.message || error));
  }
}

function finishWithError(message) {
  state.running = false;
  dom.progressOverlay.hidden = true;
  dom.errorOverlay.hidden = false;
  dom.errorText.textContent = message;
  setStatus("Run failed.", true);
  refreshRunButton();
}

async function loadResults(status) {
  showProgress("encoding", 0.97, "Loading previews…");

  const makeImage = (mode, index) => {
    const img = new Image();
    img.src = `/api/jobs/${state.jobId}/${mode === "rgb" ? "views" : "depths"}/${index}.jpg`;
    img.alt = viewLabel(index, state.angleDeg);
    img.draggable = false;
    img.hidden = mode !== state.mode;
    dom.layers.append(img);
    return img;
  };

  state.images.rgb = LAYOUT.map((_, index) => makeImage("rgb", index));
  state.images.depth = LAYOUT.map((_, index) => makeImage("depth", index));

  const everyImage = [...state.images.rgb, ...state.images.depth];
  await Promise.all(everyImage.map((img) => img.decode().catch(() => null)));

  buildSheet();
  renderInfo(status);

  dom.progressOverlay.hidden = true;
  dom.indicator.hidden = false;
  dom.sheet.hidden = false;
  dom.stage.classList.add("interactive");
  state.running = false;
  state.ready = true;
  showView(REFERENCE_VIEW);
  setMode(state.mode);
  refreshRunButton();

  dom.parallaxHint.hidden = false;
  dom.parallaxHint.style.opacity = "1";
  setTimeout(() => (dom.parallaxHint.style.opacity = "0"), 3200);

  const seconds = (status.timings.pipeline || 0).toFixed(1);
  setStatus(`Nine views in ${seconds}s. Move over the image to look around.`);
}

function buildSheet() {
  dom.sheet.innerHTML = "";
  for (const index of PHYSICAL_ORDER) {
    const thumb = document.createElement("button");
    thumb.className = "thumb" + (index === REFERENCE_VIEW ? " ref" : "");
    thumb.dataset.index = index;
    thumb.title = viewLabel(index, state.angleDeg);

    const img = new Image();
    img.src = `/api/jobs/${state.jobId}/views/${index}.jpg`;
    img.alt = "";
    const caption = document.createElement("span");
    caption.textContent = index === REFERENCE_VIEW ? `V${index} ·ref` : `V${index}`;

    thumb.append(img, caption);
    thumb.addEventListener("click", () => {
      setOrbit(false);
      setPinned(true);
      showView(index);
    });
    dom.sheet.append(thumb);
  }
}

function renderInfo(status) {
  const metadata = status.metadata || {};
  const source = status.source || {};
  const rows = [
    ["Source", `${source.filename || "—"}`],
    ["Input", `${source.original_width}×${source.original_height}`],
    ["Rendered", `${metadata.output_width}×${metadata.output_height} × 9`],
    ["Angle", `±${metadata.horizontal_angle}°`],
    ["Precision", status.precision],
    ["Model", String(metadata.model_version || "—")],
  ];
  if (source.downscaled) {
    rows.splice(2, 0, ["Downscaled to", `${source.width}×${source.height}`]);
  }

  dom.infoList.innerHTML = "";
  for (const [term, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    dom.infoList.append(dt, dd);
  }

  const pipeline = status.timings.pipeline || 0;
  const encode = status.timings.encode || 0;
  dom.timingText.textContent = `${pipeline.toFixed(1)}s + ${encode.toFixed(1)}s encode`;
  dom.validBadge.textContent = "validated";
  dom.downloadLink.href = `/api/jobs/${state.jobId}/spatial_photo.h5`;
  dom.infoCard.hidden = false;
}

/* ----------------------------------------------------------------- events */

function acceptFile(file) {
  if (!file) return;
  state.file = file;
  state.useSample = false;
  dom.dropzone.classList.add("has-file");
  dom.dropzone.querySelector(".dz-title").textContent = file.name;
  dom.dropzone.querySelector(".dz-sub").textContent =
    `${(file.size / 1024 / 1024).toFixed(1)} MB · ready`;
  state.lastRunKey = null;
  refreshRunButton();
  startJob();
}

dom.dropzone.addEventListener("click", () => dom.fileInput.click());
dom.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    dom.fileInput.click();
  }
});
dom.fileInput.addEventListener("change", (event) => acceptFile(event.target.files[0]));

for (const eventName of ["dragenter", "dragover"]) {
  document.addEventListener(eventName, (event) => {
    event.preventDefault();
    dom.dropzone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  document.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (eventName === "drop" || event.relatedTarget === null) {
      dom.dropzone.classList.remove("dragging");
    }
  });
}
document.addEventListener("drop", (event) => {
  const file = event.dataTransfer && event.dataTransfer.files[0];
  if (file) acceptFile(file);
});

dom.sampleBtn.addEventListener("click", () => {
  state.useSample = true;
  state.file = null;
  dom.dropzone.classList.add("has-file");
  dom.dropzone.querySelector(".dz-title").textContent = "teaser.jpg";
  dom.dropzone.querySelector(".dz-sub").textContent = "bundled sample · ready";
  state.lastRunKey = null;
  refreshRunButton();
  startJob();
});

dom.runBtn.addEventListener("click", startJob);
dom.angle.addEventListener("input", () => {
  updateSliderFill();
  refreshRunButton();
});
dom.maxSize.addEventListener("change", refreshRunButton);
dom.precision.addEventListener("change", refreshRunButton);

document.querySelectorAll(".seg").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

dom.orbitBtn.addEventListener("click", () => {
  if (!state.ready) return;
  setPinned(false);
  setOrbit(!state.orbit);
});

dom.pinBtn.addEventListener("click", () => {
  if (!state.ready) return;
  setOrbit(false);
  setPinned(!state.pinned);
});

dom.stage.addEventListener("pointermove", (event) => {
  if (!state.ready || state.pinned || state.orbit) return;
  const index = indexFromPointer(event);
  if (index !== state.index) showView(index);
});

dom.stage.addEventListener("pointerleave", () => {
  if (!state.ready || state.pinned || state.orbit) return;
  showView(REFERENCE_VIEW);
});

dom.stage.addEventListener("click", () => {
  if (!state.ready) return;
  setOrbit(false);
  setPinned(!state.pinned);
});

document.addEventListener("keydown", (event) => {
  if (!state.ready || event.target.matches("input, select, textarea")) return;
  const [h, v] = LAYOUT[state.index];
  let dh = 0;
  let dv = 0;
  if (event.key === "ArrowLeft") dh = -1;
  else if (event.key === "ArrowRight") dh = +1;
  else if (event.key === "ArrowUp") dv = -1; // up == negative vertical sign
  else if (event.key === "ArrowDown") dv = +1;
  else if (event.key.toLowerCase() === "d") return setMode(state.mode === "rgb" ? "depth" : "rgb");
  else if (event.key.toLowerCase() === "o") {
    setPinned(false);
    return setOrbit(!state.orbit);
  } else return;

  event.preventDefault();
  setOrbit(false);
  setPinned(true);
  const next = LAYOUT.findIndex(
    ([lh, lv]) => lh === clamp(h + dh, -1, 1) && lv === clamp(v + dv, -1, 1)
  );
  if (next >= 0) showView(next);
});

for (let cell = 0; cell < 9; cell += 1) {
  const dot = document.createElement("i");
  if (PHYSICAL_ORDER[cell] === REFERENCE_VIEW) dot.classList.add("ref");
  dom.indicator.append(dot);
}

updateSliderFill();
setPinned(false);
loadConfig();
