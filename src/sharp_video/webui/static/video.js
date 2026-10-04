"use strict";

/**
 * Stage 1's camera layout (sharp_spatialize.cameras.DEFAULT_LAYOUT) as
 * (horizontal, vertical) offset signs -- the same constants as the photo page.
 * OpenCV convention: +Y is down, so a +1 vertical sign puts that camera BELOW
 * the reference view.
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
const POLL_INTERVAL_MS = 500;

const STAGE_TEXT = {
  uploading: "Uploading…",
  queued: "Waiting for another preview to finish…",
  probing: "Reading the MIV bitstream…",
  decoding: "Decoding with TmivDecoder…",
  encoding: "Building the 3×3 preview videos…",
  loading: "Loading the preview…",
  done: "Done",
  // the video -> MIV conversion, which runs before any of the above
  reading: "Reading the video…",
  spatializing: "Spatializing frames with SHARP…",
  encoding_miv: "Encoding MIV with TMIV…",
  cancelled: "Stopped",
};

/**
 * Conversion stages, worded for a run that takes hours rather than seconds.
 * STAGE_TEXT's "queued" is about previews waiting on the decoder, not this.
 */
const CONVERSION_STAGE_TEXT = {
  queued: "Waiting for the conversion ahead of this one…",
  reading: "Reading the video…",
  spatializing: "Spatializing frames with SHARP…",
  encoding_miv: "Encoding MIV with TMIV…",
};

/** Containers the picker offers until the server says what it takes. */
const FALLBACK_VIDEO_SUFFIXES = [".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"];

/** A conversion is still worth polling in these states. */
const LIVE = new Set(["queued", "running"]);

const el = (id) => document.getElementById(id);

const dom = {
  dropzone: el("dropzone"),
  fileInput: el("fileInput"),
  dzFormats: el("dzFormats"),
  convertCard: el("convertCard"),
  convertFile: el("convertFile"),
  convertClear: el("convertClear"),
  convertBtn: el("convertBtn"),
  optFrames: el("optFrames"),
  optStart: el("optStart"),
  optHeight: el("optHeight"),
  optAngle: el("optAngle"),
  optAngleOut: el("optAngleOut"),
  optQuality: el("optQuality"),
  conversionsField: el("conversionsField"),
  conversions: el("conversions"),
  stopBtn: el("stopBtn"),
  errorTitle: el("errorTitle"),
  library: el("library"),
  libraryHint: el("libraryHint"),
  refreshBtn: el("refreshBtn"),
  statusLine: el("statusLine"),
  tmivChip: el("tmivChip"),
  stage: el("stage"),
  viewport: el("viewport"),
  colorVideo: el("colorVideo"),
  depthVideo: el("depthVideo"),
  emptyState: el("emptyState"),
  progressOverlay: el("progressOverlay"),
  progressStage: el("progressStage"),
  progressBar: el("progressBar"),
  progressNote: el("progressNote"),
  errorOverlay: el("errorOverlay"),
  errorText: el("errorText"),
  indicator: el("indicator"),
  parallaxHint: el("parallaxHint"),
  viewLabel: el("viewLabel"),
  orbitBtn: el("orbitBtn"),
  pinBtn: el("pinBtn"),
  transport: el("transport"),
  playBtn: el("playBtn"),
  prevBtn: el("prevBtn"),
  nextBtn: el("nextBtn"),
  scrubber: el("scrubber"),
  frameText: el("frameText"),
  speed: el("speed"),
  infoCard: el("infoCard"),
  infoList: el("infoList"),
  timingText: el("timingText"),
  downloadLink: el("downloadLink"),
};

const state = {
  library: null,
  sourceId: null,
  pendingVideo: null, // the video chosen in the picker, not yet uploaded
  conversionId: null, // the conversion the overlay is following, if any
  meta: null,
  token: 0, // bumped per selection; stale async work sees the change and stops
  ready: false,
  mode: "rgb",
  layout: "single",
  index: REFERENCE_VIEW,
  pinned: false,
  orbit: false,
  orbitTimer: null,
  orbitStep: 0,
  blobUrls: [],
  frame: -1,
  scrubbing: false,
};

/* ---------------------------------------------------------------- helpers */

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function fmtBytes(bytes) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} kB`;
}

/** Seconds as something a person can plan around: "2 h 20 m", "7 m 30 s", "45 s". */
function fmtDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  if (seconds >= 3600) {
    return `${Math.floor(seconds / 3600)} h ${Math.round((seconds % 3600) / 60)} m`;
  }
  if (seconds >= 60) return `${Math.floor(seconds / 60)} m ${Math.round(seconds % 60)} s`;
  return `${Math.round(seconds)} s`;
}

function fmtFps(fps) {
  return Number.isInteger(fps) ? String(fps) : Number(fps).toFixed(3).replace(/0+$/, "");
}

function viewLabel(index) {
  const [h, v] = LAYOUT[index];
  if (h === 0 && v === 0) return `V${index} · reference view`;
  const parts = [];
  if (v !== 0) parts.push(v < 0 ? "up" : "down");
  if (h !== 0) parts.push(h < 0 ? "left" : "right");
  return `V${index} · ${parts.join("-")}`;
}

function setStatus(message, isError = false) {
  dom.statusLine.textContent = message || "";
  dom.statusLine.classList.toggle("error", isError);
}

function mosaicOrder() {
  return (state.meta && state.meta.mosaic_order) || PHYSICAL_ORDER;
}

/* ---------------------------------------------------------------- videos */

const videos = () => [dom.colorVideo, dom.depthVideo];
const activeVideo = () => (state.mode === "rgb" ? dom.colorVideo : dom.depthVideo);
const frameCount = () => state.meta.frame_count;

function frameOf(video) {
  return clamp(Math.floor(video.currentTime * state.meta.fps + 1e-3), 0, frameCount() - 1);
}

/** Seek into the middle of a frame, so rounding can never show its neighbour. */
function timeOf(frame) {
  return (frame + 0.5) / state.meta.fps;
}

function seekVideo(video, time) {
  return new Promise((resolve) => {
    if (Math.abs(video.currentTime - time) < 1e-4) return resolve();
    video.addEventListener("seeked", () => resolve(), { once: true });
    video.currentTime = time;
  });
}

function waitForData(video) {
  return new Promise((resolve, reject) => {
    if (video.readyState >= 2) return resolve();
    const onError = () => reject(new Error("The browser could not decode the preview video."));
    video.addEventListener("loadeddata", () => {
      video.removeEventListener("error", onError);
      resolve();
    }, { once: true });
    video.addEventListener("error", onError, { once: true });
  });
}

/** GET a file into a blob URL, reporting progress; blob URLs seek reliably. */
async function fetchBlob(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      message = (await response.json()).error || message;
    } catch (_) { /* not JSON */ }
    throw new Error(message);
  }
  const total = Number(response.headers.get("Content-Length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) onProgress(received / total);
  }
  return URL.createObjectURL(new Blob(chunks, { type: "video/mp4" }));
}

/* ------------------------------------------------------------- view state */

function showView(index) {
  state.index = index;
  const cell = mosaicOrder().indexOf(index);
  dom.viewport.style.setProperty("--col", cell % 3);
  dom.viewport.style.setProperty("--row", Math.floor(cell / 3));
  dom.viewLabel.textContent = viewLabel(index);
  Array.from(dom.indicator.children).forEach((dot, position) => {
    dot.classList.toggle("on", mosaicOrder()[position] === index);
  });
}

async function setMode(mode) {
  if (mode === state.mode) return;
  const from = activeVideo();
  state.mode = mode;
  const to = activeVideo();
  document.querySelectorAll(".seg[data-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  if (!state.ready) return;

  // Only the visible layer plays; the other catches up to the same frame first.
  const playing = !from.paused;
  from.pause();
  await seekVideo(to, from.currentTime);
  if (state.mode !== mode) return; // toggled again while seeking
  to.classList.add("active");
  from.classList.remove("active");
  if (playing) to.play().catch(() => {});
  updatePlayButton();
}

function setLayout(layout) {
  state.layout = layout;
  dom.viewport.classList.toggle("single", layout === "single");
  dom.viewport.classList.toggle("grid", layout === "grid");
  dom.stage.classList.toggle("grid-layout", layout === "grid");
  document.querySelectorAll(".seg[data-layout]").forEach((button) => {
    button.classList.toggle("active", button.dataset.layout === layout);
  });
  if (layout === "grid") setPinned(false);
  fitViewport();
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

function setControlsEnabled(enabled) {
  for (const button of [dom.orbitBtn, dom.pinBtn]) button.disabled = !enabled;
}

/** Size the viewport to the largest box of the tile's aspect that fits the stage. */
function fitViewport() {
  if (!state.meta) return;
  const [tileW, tileH] = state.meta.tile_size;
  const bounds = dom.stage.getBoundingClientRect();
  const scale = Math.min(bounds.width / tileW, bounds.height / tileH);
  dom.viewport.style.width = `${Math.floor(tileW * scale)}px`;
  dom.viewport.style.height = `${Math.floor(tileH * scale)}px`;
}

/** The view whose tile is under the pointer, treating `element` as the 3x3 mosaic. */
function viewUnderPointer(event, element) {
  const bounds = element.getBoundingClientRect();
  const fx = (event.clientX - bounds.left) / bounds.width;
  const fy = (event.clientY - bounds.top) / bounds.height;
  if (fx < 0 || fx >= 1 || fy < 0 || fy >= 1) return null;
  return mosaicOrder()[Math.floor(fy * 3) * 3 + Math.floor(fx * 3)];
}

/* -------------------------------------------------------------- transport */

function updatePlayButton() {
  const playing = state.ready && !activeVideo().paused;
  dom.playBtn.textContent = playing ? "❚❚" : "▶";
  dom.playBtn.setAttribute("aria-label", playing ? "Pause (space)" : "Play (space)");
}

function togglePlay() {
  if (!state.ready) return;
  const video = activeVideo();
  if (video.paused) video.play().catch(() => {});
  else video.pause();
}

async function goToFrame(frame) {
  if (!state.ready) return;
  const video = activeVideo();
  video.pause();
  await seekVideo(video, timeOf(clamp(frame, 0, frameCount() - 1)));
}

function updateScrubberFill() {
  const max = Number(dom.scrubber.max) || 1;
  dom.scrubber.style.setProperty("--fill", `${(Number(dom.scrubber.value) / max) * 100}%`);
}

/** Per display frame: keep the frame counter and scrubber on the video's frame. */
function tick() {
  if (state.ready) {
    const frame = frameOf(activeVideo());
    if (frame !== state.frame) {
      state.frame = frame;
      if (!state.scrubbing) dom.scrubber.value = frame;
      updateScrubberFill();
      const seconds = state.meta.timestamps[frame] ?? frame / state.meta.fps;
      dom.frameText.textContent = `${frame + 1} / ${frameCount()} · ${seconds.toFixed(2)} s`;
    }
  }
  requestAnimationFrame(tick);
}

/* -------------------------------------------------------------- library */

async function loadLibrary() {
  try {
    const response = await fetch("/api/miv/library");
    const library = await response.json();
    if (!response.ok) throw new Error(library.error || `HTTP ${response.status}`);
    state.library = library;
    renderTmiv(library.tmiv);
    renderConverter(library.convert);
    renderConversions(library.conversions || []);
    renderLibrary();
    return library;
  } catch (error) {
    dom.tmivChip.textContent = "server unreachable";
    dom.tmivChip.className = "chip warn";
    return null;
  }
}

function renderTmiv(tmiv) {
  dom.tmivChip.className = `chip ${tmiv.ok ? "ok" : "warn"}`;
  dom.tmivChip.textContent = tmiv.ok ? `TMIV ${tmiv.version || ""}`.trim() : "TMIV not found";
  dom.tmivChip.title = tmiv.ok ? tmiv.root : tmiv.error;
  if (!tmiv.ok) setStatus(`Converting and previewing both need TMIV: ${tmiv.error}`, true);
}

/** Teach the picker and the conversion form what this server accepts. */
function renderConverter(convert) {
  if (!convert) return;
  const suffixes = convert.suffixes || FALLBACK_VIDEO_SUFFIXES;
  dom.fileInput.accept = [".miv", ...suffixes].join(",");
  dom.dzFormats.textContent = `${suffixes.slice(0, 3).map((s) => s.slice(1)).join(", ")}… → MIV`;
  if (dom.optHeight.options.length) return; // the form keeps what the user set

  for (const height of convert.view_heights || [null, 1080, 720, 540, 360]) {
    const option = document.createElement("option");
    option.value = height === null ? "0" : String(height);
    option.textContent = height === null ? "Source size" : `${height}p`;
    option.selected = height === convert.defaults.view_height;
    dom.optHeight.append(option);
  }
  // An empty frame count means the whole video, which is the default.
  dom.optFrames.value = convert.defaults.max_frames ?? "";
  dom.optStart.value = convert.defaults.start_frame;
  if (convert.min_angle_deg) dom.optAngle.min = convert.min_angle_deg;
  if (convert.max_angle_deg) dom.optAngle.max = convert.max_angle_deg;
  dom.optAngle.value = convert.defaults.angle_deg;
  updateAngleSlider();
}

/** Fill the angle slider and show its degrees, exactly as the photo page does. */
function updateAngleSlider() {
  const input = dom.optAngle;
  const fraction = (input.value - input.min) / (input.max - input.min);
  input.style.setProperty("--fill", `${fraction * 100}%`);
  dom.optAngleOut.textContent = `${Number(input.value)}°`;
}

function conversionDetails(conversion) {
  const parts = [];
  if (conversion.state === "done") {
    parts.push(`${conversion.miv.frame_count} frames`, fmtBytes(conversion.miv.size_bytes));
  } else if (conversion.frame_count) {
    parts.push(`${conversion.frames_done} / ${conversion.frame_count} frames`);
  }
  if (conversion.state === "error") parts.push("see the message");
  parts.push(fmtDuration(conversion.elapsed_s));
  return parts.join(" · ");
}

const CONVERSION_TAG = {
  done: ["converted", "ready"],
  error: ["failed", "failed"],
  cancelled: ["stopped", ""],
};

/** The conversions this server remembers, newest first; only while there are any. */
function renderConversions(conversions) {
  dom.conversionsField.hidden = conversions.length === 0;
  dom.conversions.innerHTML = "";
  for (const conversion of conversions) {
    const [tagText, tagClass] = CONVERSION_TAG[conversion.state]
      || [conversion.cancelling ? "stopping" : "converting", "busy"];
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "lib-item";
    button.title = conversion.error || conversion.name;

    const name = document.createElement("span");
    name.className = "lib-name";
    name.textContent = conversion.name;
    const tag = document.createElement("span");
    tag.className = `lib-tag ${tagClass}`;
    tag.textContent = tagText;
    const sub = document.createElement("span");
    sub.className = "lib-sub";
    sub.textContent = conversionDetails(conversion);

    button.append(name, tag, sub);
    button.addEventListener("click", () => {
      if (conversion.state === "done") return selectSource(conversion.source_id);
      if (LIVE.has(conversion.state)) return watchConversion(conversion.id);
      showError(conversion.error || "The conversion did not finish.",
                conversion.state === "cancelled" ? "That conversion was stopped"
                                                 : "That conversion failed");
    });
    item.append(button);
    dom.conversions.append(item);
  }
}

function sourceDetails(source) {
  const parts = [];
  if (source.frame_count) parts.push(`${source.frame_count} frames`);
  if (source.fps) parts.push(`${fmtFps(source.fps)} fps`);
  if (source.resolution) parts.push(`${source.resolution[0]}×${source.resolution[1]}`);
  parts.push(fmtBytes(source.size_bytes));
  if (!source.has_manifest) parts.push("no manifest");
  return parts.join(" · ");
}

function sourceTag(source) {
  const preview = source.preview;
  if (preview && preview.state === "done") return ["ready", "ready"];
  if (preview && (preview.state === "running" || preview.state === "queued")) {
    return ["building", "busy"];
  }
  if (preview && preview.state === "error") return ["failed", "failed"];
  return [{ upload: "uploaded", converted: "converted" }[source.origin] || "", ""];
}

function renderLibrary() {
  const library = state.library;
  if (!library) return;
  dom.library.innerHTML = "";
  for (const source of library.sources) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "lib-item" + (source.id === state.sourceId ? " active" : "");
    button.title = source.path;

    const name = document.createElement("span");
    name.className = "lib-name";
    name.textContent = source.name;
    const [tagText, tagClass] = sourceTag(source);
    const tag = document.createElement("span");
    tag.className = `lib-tag ${tagClass}`;
    tag.textContent = tagText;
    const sub = document.createElement("span");
    sub.className = "lib-sub";
    sub.textContent = sourceDetails(source);

    button.append(name, tag, sub);
    button.addEventListener("click", () => selectSource(source.id));
    item.append(button);
    dom.library.append(item);
  }
  const roots = library.roots.join(", ");
  dom.libraryHint.textContent = library.sources.length
    ? `*.miv in ${roots}. Previews are cached in ${library.cache_dir}.`
    : `No .miv files in ${roots}. Drop one above, or restart the server with the folder ` +
      "that holds your clips, e.g. sharp-video-webui ~/out";
}

/* ------------------------------------------------------------ the preview */

function showProgress(stage, progress, note = "", label = null) {
  dom.emptyState.hidden = true;
  dom.errorOverlay.hidden = true;
  dom.progressOverlay.hidden = false;
  dom.progressStage.textContent = label || STAGE_TEXT[stage] || stage;
  dom.progressBar.style.width = `${clamp(progress, 0.02, 1) * 100}%`;
  dom.progressNote.textContent = note;
}

function showError(message, title = "That preview failed") {
  dom.progressOverlay.hidden = true;
  dom.errorOverlay.hidden = false;
  dom.errorTitle.textContent = title;
  dom.errorText.textContent = message;
  setStatus(`${title}.`, true);
}

/** Show the stop button while a conversion of ours is running, and nowhere else. */
function followingConversion(id) {
  state.conversionId = id;
  dom.stopBtn.hidden = id === null;
  dom.stopBtn.disabled = false;
  dom.stopBtn.textContent = "Stop converting";
}

function resetViewer() {
  followingConversion(null);
  setOrbit(false);
  setPinned(false);
  setControlsEnabled(false);
  state.ready = false;
  state.meta = null;
  state.frame = -1;
  for (const video of videos()) {
    video.pause();
    video.removeAttribute("src");
    video.load();
    video.classList.remove("active");
  }
  for (const url of state.blobUrls) URL.revokeObjectURL(url);
  state.blobUrls = [];
  dom.stage.classList.remove("interactive");
  for (const node of [dom.viewport, dom.transport, dom.indicator, dom.infoCard,
    dom.parallaxHint, dom.errorOverlay, dom.emptyState]) {
    node.hidden = true;
  }
  dom.viewLabel.textContent = "—";
  setStatus("");
}

async function selectSource(id) {
  const token = ++state.token;
  state.sourceId = id;
  history.replaceState(null, "", `/video?source=${encodeURIComponent(id)}`);
  resetViewer();
  renderLibrary();
  showProgress("queued", 0.02);
  try {
    const response = await fetch(`/api/miv/sources/${id}/preview`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    await followPreview(id, token);
  } catch (error) {
    if (token === state.token) showError(String(error.message || error));
  }
}

async function followPreview(id, token) {
  while (token === state.token) {
    const response = await fetch(`/api/miv/previews/${id}`);
    const status = await response.json();
    if (!response.ok) throw new Error(status.error || `HTTP ${response.status}`);
    if (token !== state.token) return;
    if (status.state === "error") {
      loadLibrary();
      throw new Error(status.error);
    }
    if (status.state === "done") return loadPreview(id, status.meta, token);
    const note = status.frame_count && (status.stage === "decoding" || status.stage === "encoding")
      ? `frame ${status.frames_done} of ${status.frame_count}`
      : "";
    showProgress(status.stage, status.progress, note);
    await sleep(POLL_INTERVAL_MS);
  }
}

async function loadPreview(id, meta, token) {
  const urls = [];
  try {
    for (const [position, layer] of ["color", "depth"].entries()) {
      urls.push(await fetchBlob(`/api/miv/previews/${id}/${layer}.mp4`, (fraction) => {
        if (token !== state.token) return;
        const overall = (position + fraction) / 2;
        showProgress("loading", overall, `${layer} video · ${Math.round(fraction * 100)}%`);
      }));
      if (token !== state.token) return;
    }
  } finally {
    if (token !== state.token) urls.forEach((url) => URL.revokeObjectURL(url));
  }

  state.meta = meta;
  state.blobUrls = urls;
  // Lay the videos out (still under the progress overlay) before they load:
  // browsers may not decode media inside a display:none subtree.
  dom.viewport.hidden = false;
  setLayout(state.layout);
  [dom.colorVideo.src, dom.depthVideo.src] = urls;
  const rate = Number(dom.speed.value);
  for (const video of videos()) video.playbackRate = rate;
  await Promise.all(videos().map(waitForData));
  if (token !== state.token) return;

  dom.scrubber.max = String(meta.frame_count - 1);
  dom.scrubber.value = "0";
  activeVideo().classList.add("active");
  dom.progressOverlay.hidden = true;
  dom.transport.hidden = false;
  dom.indicator.hidden = false;
  dom.stage.classList.add("interactive");
  setControlsEnabled(true);
  state.ready = true;
  setLayout(state.layout);
  showView(REFERENCE_VIEW);
  renderInfo(id, meta);
  activeVideo().play().catch(() => {});
  updatePlayButton();

  dom.parallaxHint.hidden = false;
  dom.parallaxHint.style.opacity = "1";
  setTimeout(() => (dom.parallaxHint.style.opacity = "0"), 3200);

  const seconds = (meta.frame_count / meta.fps).toFixed(2);
  setStatus(`${meta.frame_count} frames × 9 views (${seconds} s). Move over the video to look around.`);
  loadLibrary();
}

function renderInfo(id, meta) {
  const source = meta.source;
  const manifest = meta.manifest || {};
  const report = meta.report || {};
  const duration = meta.frame_count / meta.fps;
  const [near, far] = meta.depth_range_m;
  const rows = [
    ["File", source.name],
    ["Frames", `${meta.frame_count} @ ${fmtFps(meta.fps)} fps`],
    ["Duration", `${duration.toFixed(2)} s`],
    ["Views", `9 × ${meta.view_size[0]}×${meta.view_size[1]}`],
    ["File size", fmtBytes(source.size_bytes)],
    ["Bitrate", `${((source.size_bytes * 8) / duration / 1e6).toFixed(1)} Mbit/s`],
  ];
  if (manifest.qp) rows.push(["QP texture / depth", `${manifest.qp.texture} / ${manifest.qp.geometry}`]);
  rows.push(["Camera updates", `${meta.camera_update_frames.length} of ${meta.frame_count} frames`]);
  rows.push(["Depth in scene", `${near.toFixed(1)}–${far.toFixed(1)} m`]);
  if (report.source_video) rows.push(["Source video", report.source_video]);
  if (report.stage1_s_per_frame) {
    rows.push(["Stage 1 / frame", `${report.stage1_s_per_frame.toFixed(1)} s`]);
  }

  dom.infoList.innerHTML = "";
  for (const [term, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    dd.title = value;
    dom.infoList.append(dt, dd);
  }
  const timings = meta.timings || {};
  dom.timingText.textContent = timings.decode_s !== undefined
    ? `decode ${timings.decode_s.toFixed(0)}s · build ${timings.encode_s.toFixed(0)}s`
    : "";
  dom.downloadLink.href = `/api/miv/previews/${id}/color.mp4?download=1`;
  dom.infoCard.hidden = false;
}

async function uploadFile(file) {
  const token = ++state.token;
  resetViewer();
  hideConvertCard();
  showProgress("uploading", 0.02, `${file.name} · ${fmtBytes(file.size)}`);
  try {
    const response = await fetch(`/api/miv/upload?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      body: file,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    state.sourceId = payload.source.id;
    history.replaceState(null, "", `/video?source=${encodeURIComponent(state.sourceId)}`);
    await loadLibrary();
    await followPreview(state.sourceId, token);
  } catch (error) {
    if (token === state.token) showError(String(error.message || error));
  }
}

/* ------------------------------------------------------- video -> MIV */

function videoSuffixes() {
  const convert = state.library && state.library.convert;
  return (convert && convert.suffixes) || FALLBACK_VIDEO_SUFFIXES;
}

/** Route a chosen file: a .miv is previewed straight away, a video is converted first. */
function chooseFile(file) {
  if (!file) return;
  const name = file.name.toLowerCase();
  if (name.endsWith(".miv")) return uploadFile(file);
  if (videoSuffixes().some((suffix) => name.endsWith(suffix))) return showConvertCard(file);
  setStatus(`${file.name} is neither a .miv nor a video this server converts `
    + `(${videoSuffixes().join(", ")}).`, true);
}

function showConvertCard(file) {
  state.pendingVideo = file;
  dom.dropzone.classList.add("has-file");
  dom.convertFile.textContent = `${file.name} · ${fmtBytes(file.size)}`;
  dom.convertCard.hidden = false;
  dom.convertCard.scrollIntoView({ block: "nearest", behavior: "smooth" });
  setStatus("");
}

function hideConvertCard() {
  state.pendingVideo = null;
  dom.convertCard.hidden = true;
  dom.dropzone.classList.remove("has-file");
}

/** The conversion form as query parameters; an empty Frames box means "all of it". */
function conversionQuery(file) {
  const [qpTexture, qpGeometry] = dom.optQuality.value.split(",");
  const params = new URLSearchParams({
    filename: file.name,
    start_frame: dom.optStart.value || "0",
    max_frames: dom.optFrames.value.trim(),
    view_height: dom.optHeight.value,
    angle_deg: dom.optAngle.value,
    qp_texture: qpTexture,
    qp_geometry: qpGeometry,
  });
  return params.toString();
}

async function startConversion() {
  const file = state.pendingVideo;
  if (!file) return;
  const token = ++state.token;
  resetViewer();
  hideConvertCard();
  showProgress("uploading", 0.02, `${file.name} · ${fmtBytes(file.size)}`);
  try {
    const response = await fetch(`/api/video/upload?${conversionQuery(file)}`,
                                 { method: "POST", body: file });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    await loadLibrary(); // so the new run is listed under Conversions straight away
    if (token !== state.token) return;
    await followConversion(payload.conversion.id, token);
  } catch (error) {
    if (token === state.token) showError(String(error.message || error),
                                         "That conversion failed");
    loadLibrary();
  }
}

/** Reattach the viewer to a conversion already running on the server. */
async function watchConversion(id) {
  const token = ++state.token;
  resetViewer();
  showProgress("queued", 0.02);
  try {
    await followConversion(id, token);
  } catch (error) {
    if (token === state.token) showError(String(error.message || error),
                                         "That conversion failed");
  }
}

/** Poll one conversion to the end, then preview the .miv it wrote. */
async function followConversion(id, token) {
  followingConversion(id);
  while (token === state.token) {
    const response = await fetch(`/api/video/conversions/${id}`);
    const status = await response.json();
    if (!response.ok) throw new Error(status.error || `HTTP ${response.status}`);
    if (token !== state.token) return;

    if (status.state === "done") {
      followingConversion(null);
      await loadLibrary();
      if (token !== state.token) return;
      return selectSource(status.source_id);
    }
    if (status.state === "error" || status.state === "cancelled") {
      followingConversion(null);
      loadLibrary();
      return showError(status.error || "The conversion did not finish.",
                       status.state === "cancelled" ? "That conversion was stopped"
                                                    : "That conversion failed");
    }
    showProgress(status.stage, status.progress, conversionNote(status),
                 CONVERSION_STAGE_TEXT[status.stage]);
    patchConversion(status);
    await sleep(POLL_INTERVAL_MS);
  }
}

/** Keep the sidebar entry in step with the conversion the overlay is polling. */
function patchConversion(status) {
  if (!state.library) return;
  const conversions = state.library.conversions || (state.library.conversions = []);
  const position = conversions.findIndex((conversion) => conversion.id === status.id);
  if (position < 0) conversions.unshift(status); // started since the last library load
  else conversions[position] = status;
  renderConversions(conversions);
}

/**
 * What the progress overlay says under the bar. Stage 1 runs at seconds to
 * minutes per frame, so on a long clip the bar barely moves: the frame in
 * flight, the measured rate and the time left are what show it is working.
 */
function conversionNote(status) {
  if (status.cancelling) return "stopping after the current frame…";
  if (status.stage === "queued") return `${status.name} · next in line`;
  if (status.stage === "spatializing" && status.frame_count) {
    const done = status.frames_done;
    const parts = [`frame ${Math.min(done + 1, status.frame_count)} of ${status.frame_count}`];
    if (status.resumed_from) parts.push(`resumed from ${status.resumed_from}`);
    // Only frames this run actually spatialized time it; the ones it resumed
    // from cost it nothing, and counting them would make the rate meaningless.
    const fresh = done - status.resumed_from;
    if (fresh > 0) {
      const perFrame = status.elapsed_s / fresh;
      parts.push(`${perFrame.toFixed(1)} s/frame`,
                 `about ${fmtDuration(perFrame * (status.frame_count - done))} left`);
    } else {
      parts.push("loading the model…"); // the first frame also pays for that
    }
    return parts.join(" · ");
  }
  if (status.stage === "encoding_miv" && status.frame_count) {
    return `${status.frame_count} frames × 9 views · TMIV and VVenC, a few minutes`;
  }
  return status.name;
}

async function stopConversion() {
  if (!state.conversionId) return;
  dom.stopBtn.disabled = true;
  dom.stopBtn.textContent = "Stopping…";
  await fetch(`/api/video/conversions/${state.conversionId}/cancel`, { method: "POST" })
    .catch(() => {});
}

/* ----------------------------------------------------------------- events */

dom.dropzone.addEventListener("click", () => dom.fileInput.click());
dom.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    dom.fileInput.click();
  }
});
dom.fileInput.addEventListener("change", (event) => {
  chooseFile(event.target.files[0]);
  event.target.value = "";
});

dom.convertCard.addEventListener("submit", (event) => {
  event.preventDefault();
  startConversion();
});
dom.convertClear.addEventListener("click", hideConvertCard);
dom.optAngle.addEventListener("input", updateAngleSlider);
dom.stopBtn.addEventListener("click", stopConversion);

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
  const files = Array.from((event.dataTransfer && event.dataTransfer.files) || []);
  // A run's .miv usually travels with its .json and report; pick the video itself.
  const chosen = files.find((file) => file.name.toLowerCase().endsWith(".miv")) || files[0];
  if (chosen) chooseFile(chosen);
});

dom.refreshBtn.addEventListener("click", loadLibrary);

document.querySelectorAll(".seg[data-mode]").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});
document.querySelectorAll(".seg[data-layout]").forEach((button) => {
  button.addEventListener("click", () => setLayout(button.dataset.layout));
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
  if (!state.ready || state.orbit) return;
  if (state.layout === "grid") {
    const index = viewUnderPointer(event, dom.viewport);
    if (index !== null && index !== state.index) showView(index);
    return;
  }
  if (state.pinned) return;
  const index = viewUnderPointer(event, dom.stage);
  if (index !== null && index !== state.index) showView(index);
});

dom.stage.addEventListener("pointerleave", () => {
  if (!state.ready || state.pinned || state.orbit || state.layout === "grid") return;
  showView(REFERENCE_VIEW);
});

dom.stage.addEventListener("click", (event) => {
  if (!state.ready) return;
  setOrbit(false);
  if (state.layout === "grid") {
    const index = viewUnderPointer(event, dom.viewport);
    if (index === null) return;
    setLayout("single");
    setPinned(true);
    showView(index);
    return;
  }
  setPinned(!state.pinned);
});

dom.playBtn.addEventListener("click", togglePlay);
dom.prevBtn.addEventListener("click", () => goToFrame(state.frame - 1));
dom.nextBtn.addEventListener("click", () => goToFrame(state.frame + 1));
for (const video of videos()) {
  video.addEventListener("play", updatePlayButton);
  video.addEventListener("pause", updatePlayButton);
}

dom.scrubber.addEventListener("pointerdown", () => (state.scrubbing = true));
dom.scrubber.addEventListener("pointerup", () => (state.scrubbing = false));
dom.scrubber.addEventListener("input", () => {
  updateScrubberFill();
  goToFrame(Number(dom.scrubber.value));
});

dom.speed.addEventListener("change", () => {
  for (const video of videos()) video.playbackRate = Number(dom.speed.value);
});

document.addEventListener("keydown", (event) => {
  if (!state.ready || event.target.matches("input, select, textarea")) return;
  const key = event.key.toLowerCase();
  if (key === " ") {
    event.preventDefault();
    return togglePlay();
  }
  if (key === ",") return goToFrame(state.frame - 1);
  if (key === ".") return goToFrame(state.frame + 1);
  if (key === "home") return goToFrame(0);
  if (key === "d") return setMode(state.mode === "rgb" ? "depth" : "rgb");
  if (key === "g") return setLayout(state.layout === "grid" ? "single" : "grid");
  if (key === "o") {
    setPinned(false);
    return setOrbit(!state.orbit);
  }

  const [h, v] = LAYOUT[state.index];
  let dh = 0;
  let dv = 0;
  if (event.key === "ArrowLeft") dh = -1;
  else if (event.key === "ArrowRight") dh = +1;
  else if (event.key === "ArrowUp") dv = -1; // up == negative vertical sign
  else if (event.key === "ArrowDown") dv = +1;
  else return;

  event.preventDefault();
  setOrbit(false);
  if (state.layout === "single") setPinned(true);
  const next = LAYOUT.findIndex(
    ([lh, lv]) => lh === clamp(h + dh, -1, 1) && lv === clamp(v + dv, -1, 1)
  );
  if (next >= 0) showView(next);
});

new ResizeObserver(fitViewport).observe(dom.stage);

for (let position = 0; position < 9; position += 1) {
  const dot = document.createElement("i");
  if (PHYSICAL_ORDER[position] === REFERENCE_VIEW) dot.classList.add("ref");
  dom.indicator.append(dot);
}

setPinned(false);
dom.emptyState.hidden = false;
requestAnimationFrame(tick);
loadLibrary().then((library) => {
  if (!library) return;
  const wanted = new URLSearchParams(location.search).get("source");
  if (wanted) {
    const match = library.sources.find((source) => source.id === wanted || source.name === wanted);
    if (match) return selectSource(match.id);
  }
  // Reloading the page during a long conversion picks it back up where it is.
  const running = (library.conversions || []).find((conversion) => LIVE.has(conversion.state));
  if (running) watchConversion(running.id);
});
