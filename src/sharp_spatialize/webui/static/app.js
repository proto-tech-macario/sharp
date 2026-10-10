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

/** Viewer modes: the reference views and depth from the .h5, then the DS Presenter. */
const MODES = ["rgb", "depth", "ds", "err"];
const MODE_TAG = {
  rgb: "", depth: "",
  ds: "DS Presenter render · from the .ds.jpg",
  err: "DS render vs reference",
};

/** DS Presenter modes the server renders; the first is the default. */
const PRESENTER_NAME = { mesh: "Mesh", points: "Points" };

const STAGE_TEXT = {
  queued: "Waiting for the GPU…",
  inference: "Running SHARP — one forward pass to 3D Gaussians…",
  rendering: "Rendering 9 views with gsplat…",
  validating: "Validating geometry and depth…",
  loading: "Reading spatial_photo.h5…",
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
  compareBtn: el("compareBtn"),
  modeTag: el("modeTag"),
  errLegend: el("errLegend"),
  viewScore: el("viewScore"),
  tabHint: el("tabHint"),
  presenterSeg: el("presenterSeg"),
  panes: { views: el("paneViews"), layers: el("paneLayers"), free: el("paneFree"), quality: el("paneQuality") },
  dsCard: el("dsCard"),
  dsBadge: el("dsBadge"),
  dsCodec: el("dsCodec"),
  dsBuildBtn: el("dsBuildBtn"),
  dsProgress: el("dsProgress"),
  dsProgressBar: el("dsProgressBar"),
  dsProgressText: el("dsProgressText"),
  dsResult: el("dsResult"),
  dsSizeTotal: el("dsSizeTotal"),
  dsSizeRatio: el("dsSizeRatio"),
  dsSizeBar: el("dsSizeBar"),
  dsSizeList: el("dsSizeList"),
  dsChecks: el("dsChecks"),
  dsDownload: el("dsDownload"),
  dsOpen: el("dsOpen"),
  dsError: el("dsError"),
  layerGrid: el("layerGrid"),
  freeX: el("freeX"), freeY: el("freeY"), freeZ: el("freeZ"),
  freeXOut: el("freeXOut"), freeYOut: el("freeYOut"), freeZOut: el("freeZOut"),
  freeReset: el("freeReset"),
  freeStage: el("freeStage"),
  freeImg: el("freeImg"),
  freeBusy: el("freeBusy"),
  freeInfo: el("freeInfo"),
  qualityCards: el("qualityCards"),
  qualityTable: el("qualityTable"),
  looBtn: el("looBtn"),
  looProgress: el("looProgress"),
  looProgressBar: el("looProgressBar"),
  looText: el("looText"),
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
  images: { rgb: [], depth: [], ds: [], err: [] },
  angleDeg: 10,
  lastRunKey: null,
  comparing: false,
  tab: "views",
  ds: null, // latest DS status from the server
  presenter: "mesh", // which DS Presenter's renders and scores are shown
  dsReady: false,
  dsPollTimer: null,
  free: { busy: false, pending: false, url: null },
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
  for (const mode of MODES) {
    state.images[mode].forEach((img, i) => img.classList.toggle("active", i === index));
  }
  dom.viewLabel.textContent = viewLabel(index, state.angleDeg);
  Array.from(dom.indicator.children).forEach((cell, position) => {
    cell.classList.toggle("on", PHYSICAL_ORDER[position] === index);
  });
  Array.from(dom.sheet.children).forEach((thumb) => {
    thumb.classList.toggle("active", Number(thumb.dataset.index) === index);
  });
  renderViewScore();
}

/** The mode actually on screen: holding "compare" shows the reference colour view. */
function visibleMode() {
  return state.comparing && (state.mode === "ds" || state.mode === "err") ? "rgb" : state.mode;
}

function applyVisibleMode() {
  const shown = visibleMode();
  for (const mode of MODES) state.images[mode].forEach((img) => (img.hidden = mode !== shown));
  const isDs = shown === "ds" || shown === "err";
  dom.modeTag.hidden = !state.ready || !(isDs || state.comparing);
  dom.modeTag.textContent = state.comparing && !isDs
    ? "Reference view (SHARP) · release to return"
    : MODE_TAG[shown] + (isDs ? ` · ${PRESENTER_NAME[state.presenter]} presenter` : "");
  dom.modeTag.classList.toggle("ref", state.comparing && !isDs);
  dom.errLegend.hidden = shown !== "err";
  renderViewScore();
}

function setMode(mode) {
  if ((mode === "ds" || mode === "err") && !state.dsReady) return;
  state.mode = mode;
  document.querySelectorAll(".seg").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  dom.compareBtn.disabled = !(state.dsReady && (mode === "ds" || mode === "err"));
  applyVisibleMode();
}

function setComparing(on) {
  if (on && dom.compareBtn.disabled) return;
  state.comparing = on;
  dom.compareBtn.classList.toggle("active", on);
  applyVisibleMode();
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
  state.images = { rgb: [], depth: [], ds: [], err: [] };
  resetDS();
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

  if (status.from_h5) {
    setStatus("Loaded 9 views from the .h5. Move over the image to look around.");
  } else {
    const seconds = (status.timings.pipeline || 0).toFixed(1);
    setStatus(`Nine views in ${seconds}s. Move over the image to look around.`);
  }

  if (state.config && state.config.ds && state.config.ds.available) {
    dom.dsCard.hidden = false;
    startDS(); // CPU only, so it can start right away
  }
}

const IMAGE_URL = {
  rgb: (i) => `views/${i}.jpg`,
  depth: (i) => `depths/${i}.jpg`,
  ds: (i) => `ds/render/${i}.jpg?presenter=${state.presenter}`,
  err: (i) => `ds/error/${i}.jpg?presenter=${state.presenter}`,
};

function makeImage(mode, index) {
  const img = new Image();
  img.src = `/api/jobs/${state.jobId}/${IMAGE_URL[mode](index)}`;
  img.alt = viewLabel(index, state.angleDeg);
  img.draggable = false;
  img.hidden = mode !== visibleMode();
  img.classList.toggle("active", index === state.index);
  dom.layers.append(img);
  return img;
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
  const rows = status.from_h5
    ? [
        ["Source", `${source.filename || "—"}`],
        ["Views", `${source.width}×${source.height} × 9`],
        ["Angle", metadata.horizontal_angle != null ? `±${metadata.horizontal_angle}°` : "—"],
        ["Model", String(metadata.model_version || "—")],
      ]
    : [
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

  const pipeline = status.timings.pipeline || status.timings.load || 0;
  const encode = status.timings.encode || 0;
  dom.timingText.textContent = `${pipeline.toFixed(1)}s + ${encode.toFixed(1)}s encode`;
  dom.validBadge.textContent = status.from_h5 ? "loaded .h5" : "validated";
  if (status.from_h5 && metadata.horizontal_angle != null) state.angleDeg = Number(metadata.horizontal_angle);
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
  const isH5 = /\.(h5|hdf5)$/i.test(file.name);
  dom.dropzone.querySelector(".dz-sub").textContent =
    `${(file.size / 1024 / 1024).toFixed(1)} MB · ${isH5 ? "spatial photo, no GPU needed" : "ready"}`;
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
  if (!state.ready || state.tab !== "views" || event.target.matches("input, select, textarea")) return;
  if (event.key === " ") {
    event.preventDefault();
    if (!event.repeat) setComparing(true);
    return;
  }
  const [h, v] = LAYOUT[state.index];
  let dh = 0;
  let dv = 0;
  if (event.key === "ArrowLeft") dh = -1;
  else if (event.key === "ArrowRight") dh = +1;
  else if (event.key === "ArrowUp") dv = -1; // up == negative vertical sign
  else if (event.key === "ArrowDown") dv = +1;
  else if (event.key.toLowerCase() === "d") return setMode(state.mode === "depth" ? "rgb" : "depth");
  else if (event.key.toLowerCase() === "r") return setMode(state.mode === "ds" ? "rgb" : "ds");
  else if (event.key.toLowerCase() === "e") return setMode(state.mode === "err" ? "rgb" : "err");
  else if (event.key.toLowerCase() === "m" && state.dsReady) {
    return setPresenter(state.presenter === "mesh" ? "points" : "mesh");
  }
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

/* ------------------------------------------------------------- DS-Image */

const DS_STAGE_TEXT = {
  starting: "Starting…",
  building: "Building the 2-layer asset…",
  writing: "Writing and reading back the .ds.jpg…",
  checking: "Round-trip and file checks…",
  rendering: "Rendering the 9 source cameras…",
  done: "Done",
};

const SIZE_PARTS = [
  ["primary_jpeg_bytes", "Primary JPEG (Layer 0 colour)", "#5aa9ff"],
  ["base_depth_bytes", "Layer 0 depth", "#ffb44d"],
  ["layer1_rgb_bytes", "Layer 1 colour", "#46d18a"],
  ["layer1_depth_bytes", "Layer 1 depth", "#c084fc"],
  ["layer1_mask_bytes", "Layer 1 mask", "#8b94a8"],
  ["other", "Metadata + container", "#3a4256"],
];

const QUALITY_ROWS = [
  ["presenter_only", "Presenter only", "2-layer asset built straight from the .h5, no file: checks the Presenter itself"],
  ["ds_image", "DS-Image file", "Rendered from the .ds.jpg: adds the JPEG / depth-codec loss"],
  ["leave_one_out", "Leave-one-out", "Asset rebuilt without the scored view: the honest novel-view score"],
  ["layer0_only", "Layer 0 only", "Layer 1 switched off: shows what the hidden layer adds"],
];

const fmtBytes = (n) => (n >= 1048576 ? `${(n / 1048576).toFixed(2)} MB` : `${(n / 1024).toFixed(0)} KB`);
const fmtPct = (v, digits = 1) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
const fmtDb = (v) => (v == null ? "—" : v > 99 ? "exact" : `${v.toFixed(1)} dB`);
const fmtSsim = (v) => (v == null ? "—" : v.toFixed(3));
const dsUrl = (path) => `/api/jobs/${state.jobId}/ds/${path}`;

function resetDS() {
  clearTimeout(state.dsPollTimer);
  state.ds = null;
  state.dsReady = false;
  state.comparing = false;
  dom.dsCard.hidden = true;
  dom.dsResult.hidden = true;
  dom.dsProgress.hidden = true;
  dom.dsError.hidden = true;
  dom.dsBadge.textContent = "not built";
  dom.dsBadge.className = "badge muted";
  dom.dsBuildBtn.disabled = false;
  dom.dsBuildBtn.textContent = "Build DS-Image";
  document.querySelectorAll(".ds-seg").forEach((b) => (b.disabled = true));
  dom.compareBtn.disabled = true;
  if (state.mode === "ds" || state.mode === "err") state.mode = "rgb";
  setTabsEnabled(false);
  setTab("views");
  dom.layerGrid.innerHTML = "";
  dom.qualityCards.innerHTML = "";
  dom.qualityTable.innerHTML = "";
  dom.viewScore.hidden = true;
}

async function startDS() {
  if (!state.jobId || (state.ds && state.ds.state === "running")) return;
  const jobId = state.jobId;
  state.dsReady = false;
  state.images.ds.concat(state.images.err).forEach((img) => img.remove());
  state.images.ds = [];
  state.images.err = [];
  document.querySelectorAll(".ds-seg").forEach((b) => (b.disabled = true));
  if (state.mode === "ds" || state.mode === "err") setMode("rgb");
  setTabsEnabled(false);
  if (state.tab !== "views") setTab("views");
  dom.dsError.hidden = true;
  dom.dsResult.hidden = true;
  dom.dsBuildBtn.disabled = true;
  dom.dsBuildBtn.textContent = "Building…";
  dom.dsBadge.textContent = "building";
  dom.dsBadge.className = "badge busy";
  showDSProgress("starting", 0.02);
  try {
    const response = await fetch(`/api/jobs/${jobId}/ds?codec=${dom.dsCodec.value}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    pollDS(jobId);
  } catch (error) {
    dsFailed(String(error.message || error));
  }
}

function showDSProgress(stage, progress) {
  dom.dsProgress.hidden = false;
  dom.dsProgressBar.style.width = `${clamp(progress, 0.02, 1) * 100}%`;
  dom.dsProgressText.textContent = DS_STAGE_TEXT[stage] || stage;
}

async function pollDS(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}/ds`);
    const status = await response.json();
    if (!response.ok) throw new Error(status.error || `HTTP ${response.status}`);
    if (state.jobId !== jobId) return;
    state.ds = status;
    if (status.state === "error") return dsFailed(status.error);
    if (status.state === "done") {
      if (!state.dsReady) await dsLoaded(status);
      else renderQuality(status);
      if (status.loo.state === "running") state.dsPollTimer = setTimeout(() => pollDS(jobId), 800);
      return;
    }
    showDSProgress(status.stage, status.progress);
    state.dsPollTimer = setTimeout(() => pollDS(jobId), POLL_INTERVAL_MS);
  } catch (error) {
    dsFailed(String(error.message || error));
  }
}

function dsFailed(message) {
  dom.dsProgress.hidden = true;
  dom.dsError.hidden = false;
  dom.dsError.textContent = message;
  dom.dsBadge.textContent = "failed";
  dom.dsBadge.className = "badge bad";
  dom.dsBuildBtn.disabled = false;
  dom.dsBuildBtn.textContent = "Retry";
  dom.looBtn.disabled = false;
}

/** (Re)load the DS render and error images of all 9 views for the current presenter. */
async function loadDsImages() {
  const old = state.images.ds.concat(state.images.err);
  const ds = LAYOUT.map((_, index) => makeImage("ds", index));
  const err = LAYOUT.map((_, index) => makeImage("err", index));
  await Promise.all([...ds, ...err].map((img) => img.decode().catch(() => null)));
  old.forEach((img) => img.remove()); // swap only once the new set is ready, so nothing flickers
  state.images.ds = ds;
  state.images.err = err;
  applyVisibleMode();
  showView(state.index);
}

async function setPresenter(presenter) {
  if (!PRESENTER_NAME[presenter] || presenter === state.presenter) return;
  state.presenter = presenter;
  document.querySelectorAll(".pseg").forEach((button) => {
    button.classList.toggle("active", button.dataset.presenter === presenter);
  });
  if (!state.dsReady) return;
  if (state.ds) renderQuality(state.ds);
  if (state.tab === "free") requestFreeRender();
  await loadDsImages();
}

async function dsLoaded(status) {
  await loadDsImages();
  state.dsReady = true;

  dom.dsProgress.hidden = true;
  dom.dsBuildBtn.disabled = false;
  dom.dsBuildBtn.textContent = "Rebuild";
  dom.dsBadge.textContent = status.roundtrip.passed && status.legacy_jpeg_ok ? "checks passed" : "check failed";
  dom.dsBadge.className = `badge ${status.roundtrip.passed && status.legacy_jpeg_ok ? "" : "bad"}`;
  document.querySelectorAll(".ds-seg").forEach((b) => (b.disabled = false));
  setTabsEnabled(true);

  renderSizes(status);
  renderLayers(status);
  renderQuality(status);
  setupFreeView(status);
  showView(state.index);
  setMode(state.mode === "rgb" ? "ds" : state.mode); // show the new render straight away
  setStatus("DS-Image ready. Compare modes with R / E, switch Mesh / Points with M, hold Space for the reference.");
  dom.dsCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderSizes(status) {
  const file = status.file;
  const total = file.ds_total_bytes;
  const parts = Object.fromEntries(SIZE_PARTS.map(([key]) => [key, file[key] || 0]));
  parts.other = total - SIZE_PARTS.slice(0, -1).reduce((sum, [key]) => sum + parts[key], 0);

  dom.dsSizeTotal.textContent = fmtBytes(total);
  dom.dsSizeRatio.textContent = status.h5_bytes
    ? `${(status.h5_bytes / total).toFixed(1)}× smaller than the 9-view .h5 (${fmtBytes(status.h5_bytes)})`
    : "";
  dom.dsSizeBar.innerHTML = "";
  dom.dsSizeList.innerHTML = "";
  for (const [key, label, color] of SIZE_PARTS) {
    const bytes = Math.max(parts[key], 0);
    const seg = document.createElement("i");
    seg.style.flexGrow = bytes;
    seg.style.background = color;
    seg.title = `${label}: ${fmtBytes(bytes)}`;
    dom.dsSizeBar.append(seg);
    const dt = document.createElement("dt");
    dt.innerHTML = `<i class="sw" style="background:${color}"></i>${label}`;
    const dd = document.createElement("dd");
    dd.textContent = `${fmtBytes(bytes)} · ${((100 * bytes) / total).toFixed(0)}%`;
    dom.dsSizeList.append(dt, dd);
  }
  const sp = status.sparsity;
  const dt = document.createElement("dt");
  dt.textContent = "Layer 1 occupancy";
  const dd = document.createElement("dd");
  dd.textContent = `${fmtPct(sp.occupancy)} of pixels`;
  dom.dsSizeList.append(dt, dd);

  const checks = [
    ["Round-trip", status.roundtrip.passed,
      `write → read gives back the same asset (Layer 0 colour ${fmtDb(status.roundtrip.layer0_rgb_psnr)}, ` +
      `depth max error ${status.roundtrip.layer0_depth_max_abs ?? "—"})`],
    ["Opens as plain JPEG", status.legacy_jpeg_ok, "a JPEG-only viewer shows the reference view"],
  ];
  dom.dsChecks.innerHTML = "";
  for (const [name, ok, detail] of checks) {
    const row = document.createElement("div");
    row.className = `check ${ok ? "ok" : "bad"}`;
    row.title = detail;
    row.textContent = `${ok ? "✓" : "✗"} ${name}`;
    dom.dsChecks.append(row);
  }
  dom.dsDownload.href = dsUrl("ds_image.jpg");
  dom.dsOpen.href = dsUrl("ds_image.jpg?inline=1");
  dom.dsResult.hidden = false;
}

function renderLayers(status) {
  const sp = status.sparsity;
  const file = status.file;
  const cards = [
    ["layer0_rgb", "Layer 0 · colour", `primary JPEG · ${fmtBytes(file.primary_jpeg_bytes)}`],
    ["layer0_depth", "Layer 0 · depth", `${status.codec} · ${fmtBytes(file.base_depth_bytes)}`],
    ["layer1_mask", "Layer 1 · where it has pixels", `${fmtPct(sp.occupancy)} of pixels · ${fmtBytes(file.layer1_mask_bytes)}`],
    ["layer1_rgb", "Layer 1 · colour", `hidden surfaces seen by the outer views · ${fmtBytes(file.layer1_rgb_bytes)}`],
    ["layer1_depth", "Layer 1 · depth", `same colour scale as Layer 0 · ${fmtBytes(file.layer1_depth_bytes)}`],
  ];
  dom.layerGrid.innerHTML = "";
  for (const [key, title, caption] of cards) {
    const figure = document.createElement("figure");
    figure.className = "layer-card";
    const img = new Image();
    img.src = dsUrl(`layers/${key}.jpg`);
    img.alt = title;
    img.addEventListener("click", () => window.open(img.src, "_blank", "noopener"));
    const cap = document.createElement("figcaption");
    cap.innerHTML = `<b>${title}</b><span>${caption}</span>`;
    figure.append(img, cap);
    dom.layerGrid.append(figure);
  }
}

function scoreFor(view, key) {
  if (!state.ds) return null;
  const row = state.ds.per_view.find((r) => r.view === view);
  return row && row[state.presenter] ? row[state.presenter][key] : null;
}

function renderViewScore() {
  const shown = visibleMode();
  const score = state.dsReady && (shown === "ds" || shown === "err") ? scoreFor(state.index, "ds_image") : null;
  dom.viewScore.hidden = !score;
  if (!score) return;
  dom.viewScore.innerHTML =
    `<b>V${state.index}</b> coverage ${fmtPct(score.coverage)} · PSNR ${fmtDb(score.psnr)} · SSIM ${fmtSsim(score.ssim)}`;
}

function renderQuality(status) {
  dom.qualityCards.innerHTML = "";
  for (const [key, title, detail] of QUALITY_ROWS) {
    const summary = status.summary[state.presenter][key];
    const card = document.createElement("div");
    card.className = `q-card${summary.coverage == null ? " empty" : ""}${key === "leave_one_out" ? " hero" : ""}`;
    card.innerHTML = `
      <div class="q-title">${title}</div>
      <div class="q-detail">${detail}</div>
      <div class="q-metrics">
        <span><em>${fmtPct(summary.coverage)}</em>coverage</span>
        <span><em>${fmtDb(summary.psnr)}</em>PSNR</span>
        <span><em>${fmtSsim(summary.ssim)}</em>SSIM</span>
        <span><em>${fmtPct(summary.depth_rel_median, 2)}</em>depth error</span>
      </div>`;
    dom.qualityCards.append(card);
  }

  const loo = status.loo;
  dom.looBtn.disabled = loo.state === "running";
  dom.looBtn.textContent = loo.state === "done" ? "Re-run leave-one-out" : "Run leave-one-out";
  dom.looProgress.hidden = loo.state !== "running";
  dom.looProgressBar.style.width = `${clamp(loo.progress, 0.02, 1) * 100}%`;
  if (loo.state === "error") dom.looText.textContent = loo.error;
  else if (loo.state === "running") dom.looText.textContent = `Rebuilding… ${Math.round(loo.progress * 8)} / 8 views`;
  else if (loo.state === "done") dom.looText.textContent = `Done in ${(status.timings.leave_one_out || 0).toFixed(0)}s.`;

  const head = `<thead><tr><th rowspan="2">View</th>${QUALITY_ROWS.map(([, t]) => `<th colspan="3">${t}</th>`).join("")}</tr>
    <tr>${QUALITY_ROWS.map(() => "<th>cov</th><th>PSNR</th><th>SSIM</th>").join("")}</tr></thead>`;
  const body = status.per_view.map((row) => {
    const cells = QUALITY_ROWS.map(([key]) => {
      const m = (row[state.presenter] || {})[key];
      return m ? `<td>${fmtPct(m.coverage)}</td><td>${fmtDb(m.psnr)}</td><td>${fmtSsim(m.ssim)}</td>` : "<td>—</td><td>—</td><td>—</td>";
    }).join("");
    return `<tr data-view="${row.view}"><td>${viewLabel(row.view, state.angleDeg)}</td>${cells}</tr>`;
  }).join("");
  dom.qualityTable.innerHTML = `${head}<tbody>${body}</tbody>`;
  dom.qualityTable.querySelectorAll("tbody tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      setTab("views");
      setOrbit(false);
      setPinned(true);
      setMode("err");
      showView(Number(tr.dataset.view));
    });
  });
}

async function startLeaveOneOut() {
  const jobId = state.jobId;
  dom.looBtn.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${jobId}/ds/loo`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    state.ds = payload;
    renderQuality(payload);
    pollDS(jobId);
  } catch (error) {
    dom.looBtn.disabled = false;
    dom.looText.textContent = String(error.message || error);
  }
}

/* ------------------------------------------------------------- free view */

function freeRange() {
  // A slider end moves the camera by a fifth of the scene's median depth.
  return 0.2 * ((state.ds && state.ds.median_depth) || 1);
}

function freeOffset() {
  const r = freeRange();
  return [Number(dom.freeX.value) * r, Number(dom.freeY.value) * r, Number(dom.freeZ.value) * r];
}

function setupFreeView() {
  for (const input of [dom.freeX, dom.freeY, dom.freeZ]) input.value = 0;
  updateFreeLabels();
  requestFreeRender();
}

function updateFreeLabels() {
  const [x, y, z] = freeOffset();
  const cm = (v) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)} cm`;
  dom.freeXOut.textContent = cm(x);
  dom.freeYOut.textContent = cm(-y); // +Y is down in OpenCV; show "up" as positive
  dom.freeZOut.textContent = cm(z);
  for (const input of [dom.freeX, dom.freeY, dom.freeZ]) {
    input.style.setProperty("--fill", `${((Number(input.value) + 1) / 2) * 100}%`);
  }
}

async function requestFreeRender() {
  if (!state.dsReady) return;
  if (state.free.busy) {
    state.free.pending = true; // latest position wins once the current render returns
    return;
  }
  state.free.busy = true;
  state.free.pending = false;
  dom.freeBusy.hidden = false;
  const [x, y, z] = freeOffset();
  const started = performance.now();
  try {
    const query = `x=${x.toFixed(4)}&y=${y.toFixed(4)}&z=${z.toFixed(4)}&presenter=${state.presenter}`;
    const response = await fetch(dsUrl(`free.jpg?${query}`));
    if (!response.ok) throw new Error((await response.json()).error || `HTTP ${response.status}`);
    const coverage = Number(response.headers.get("X-Coverage"));
    const blob = await response.blob();
    if (state.free.url) URL.revokeObjectURL(state.free.url);
    state.free.url = URL.createObjectURL(blob);
    dom.freeImg.src = state.free.url;
    const ms = performance.now() - started;
    dom.freeInfo.hidden = false;
    dom.freeInfo.innerHTML = `coverage ${fmtPct(coverage)} · rendered in ${(ms / 1000).toFixed(2)}s`;
  } catch (error) {
    dom.freeInfo.hidden = false;
    dom.freeInfo.textContent = String(error.message || error);
  } finally {
    state.free.busy = false;
    dom.freeBusy.hidden = true;
    if (state.free.pending) requestFreeRender();
  }
}

for (const input of [dom.freeX, dom.freeY, dom.freeZ]) {
  input.addEventListener("input", () => {
    updateFreeLabels();
    requestFreeRender();
  });
}

dom.freeReset.addEventListener("click", setupFreeView);

let freeDrag = null;
dom.freeStage.addEventListener("pointerdown", (event) => {
  if (!state.dsReady) return;
  freeDrag = { x: event.clientX, y: event.clientY, fx: Number(dom.freeX.value), fy: Number(dom.freeY.value) };
  dom.freeStage.setPointerCapture(event.pointerId);
});
dom.freeStage.addEventListener("pointermove", (event) => {
  if (!freeDrag) return;
  const bounds = dom.freeStage.getBoundingClientRect();
  // Dragging the scene right moves the camera left, like grabbing the photo.
  dom.freeX.value = clamp(freeDrag.fx - (2 * (event.clientX - freeDrag.x)) / bounds.width, -1, 1);
  dom.freeY.value = clamp(freeDrag.fy - (2 * (event.clientY - freeDrag.y)) / bounds.height, -1, 1);
  updateFreeLabels();
  requestFreeRender();
});
for (const name of ["pointerup", "pointercancel"]) {
  dom.freeStage.addEventListener(name, () => (freeDrag = null));
}

/* ------------------------------------------------------------------ tabs */

function setTabsEnabled(enabled) {
  document.querySelectorAll(".tab").forEach((tab) => {
    if (tab.dataset.tab !== "views") tab.disabled = !enabled;
  });
  dom.tabHint.hidden = enabled || !(state.config && state.config.ds && state.config.ds.available);
  dom.presenterSeg.hidden = !enabled;
}

function setTab(name) {
  state.tab = name;
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  for (const [key, pane] of Object.entries(dom.panes)) pane.hidden = key !== name;
  if (name !== "views") setOrbit(false);
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => !tab.disabled && setTab(tab.dataset.tab));
});

dom.dsBuildBtn.addEventListener("click", startDS);
dom.dsCodec.addEventListener("change", () => {
  if (state.dsReady) dom.dsBuildBtn.textContent = "Rebuild with this codec";
});
dom.looBtn.addEventListener("click", startLeaveOneOut);
document.querySelectorAll(".pseg").forEach((button) => {
  button.addEventListener("click", () => setPresenter(button.dataset.presenter));
});

dom.compareBtn.addEventListener("pointerdown", () => setComparing(true));
for (const name of ["pointerup", "pointerleave", "pointercancel"]) {
  dom.compareBtn.addEventListener(name, () => state.comparing && setComparing(false));
}
document.addEventListener("keyup", (event) => {
  if (event.key === " " && state.comparing) setComparing(false);
});

for (let cell = 0; cell < 9; cell += 1) {
  const dot = document.createElement("i");
  if (PHYSICAL_ORDER[cell] === REFERENCE_VIEW) dot.classList.add("ref");
  dom.indicator.append(dot);
}

updateSliderFill();
setPinned(false);
loadConfig().then(() => setTabsEnabled(false));
