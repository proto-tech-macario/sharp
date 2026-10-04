"""Browser previews of MIV files for the web UI.

A browser cannot play MIV, so the preview shows what the reference decoder
reconstructs. TmivDecoder decodes the file; every frame's 9 views are then
tiled into a 3x3 mosaic and written as two H.264 MP4s, one of texture and one
of colorized depth. Each tile sits where its camera sits (the top row holds
the cameras above the reference view), so the mosaic reads as the rig, and
the client shows a single view by cropping its tile: one video per layer keeps
all 9 views frame-locked without any syncing.

Previews are cached on disk per file version (path, size, mtime), so a clip is
decoded once. TmivDecoder's raw output (about 50 MB per frame at 720p) is
deleted as soon as the mosaics are written.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO

import av
import numpy as np
from PIL import Image
from sharp_spatialize.webui.jobs import INVALID_DEPTH_COLOR, turbo

from ..miv.decode import decode_miv, decoded_frames_written, probe_frame_count
from ..miv.tmiv import TmivInstall, run_logged

#: Bump when the preview files change shape, so stale caches are rebuilt.
PREVIEW_VERSION = 1

#: Mosaic cell (row-major, top-left first) -> view index. Stage 1's layout puts
#: v0..v2 below the reference view (OpenCV +Y is down), so the top row is v6..v8.
MOSAIC_ORDER = (6, 7, 8, 3, 4, 5, 0, 1, 2)

#: Largest mosaic written: 3x3 tiles of 1280x720, inside what browsers
#: hardware-decode for H.264. Larger views are scaled down to fit.
MAX_MOSAIC_SIZE = (3840, 2160)

LAYERS = ("color", "depth")
PREVIEW_FILES = tuple(f"{layer}.mp4" for layer in LAYERS)

#: Keyframe interval in frames: short, so scrubbing and frame stepping stay quick.
GOP_FRAMES = 10
H264_CRF = 18

#: Share of the progress bar owned by decoding; building the mosaics fills the rest.
DECODE_SHARE = 0.5

LOG_TAIL_LINES = 12
CHUNK_BYTES = 1 << 20


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ sources


#: Where a .miv in the library came from: named on the command line, uploaded
#: as a .miv, or converted from an ordinary video by the web UI itself.
ORIGINS = ("file", "upload", "converted")


@dataclass(frozen=True)
class MivSource:
    """One .miv file on offer, identified by its path, size and modification time."""

    path: Path
    size_bytes: int
    mtime_ns: int
    origin: str = "file"

    @classmethod
    def from_path(cls, path: str | Path, origin: str = "file") -> MivSource:
        """Describe the file at `path` as it is now."""
        path = Path(path).resolve()
        stat = path.stat()
        return cls(path, stat.st_size, stat.st_mtime_ns, origin)

    @property
    def name(self) -> str:
        """The file name shown in the library."""
        return self.path.name

    @property
    def uploaded(self) -> bool:
        """Whether the file arrived through the browser rather than off the command line."""
        return self.origin == "upload"

    @property
    def source_id(self) -> str:
        """Stable while the file is unchanged; a re-encoded file gets a new id and preview."""
        key = f"{self.path}\0{self.size_bytes}\0{self.mtime_ns}"
        return hashlib.sha1(key.encode()).hexdigest()[:12]

    def manifest(self) -> dict | None:
        """`<file>.miv.json`, written by the Stage 2 encoder, if it is there."""
        return _read_json(Path(f"{self.path}.json"))

    def report(self) -> dict | None:
        """`<file>.miv.report.json`, written by `sharp_video_to_miv`, if it is there."""
        return _read_json(Path(f"{self.path}.report.json"))

    def describe(self) -> dict[str, Any]:
        """What the library list shows."""
        manifest = self.manifest() or {}
        return {
            "id": self.source_id,
            "name": self.name,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "modified": self.mtime_ns / 1e9,
            "origin": self.origin,
            "uploaded": self.uploaded,
            "has_manifest": bool(manifest),
            "frame_count": manifest.get("frame_count"),
            "fps": manifest.get("fps"),
            "resolution": manifest.get("resolution"),
        }


def store_upload(directory: str | Path, filename: str, stream: BinaryIO, length: int) -> Path:
    """Write `length` bytes from `stream` to `<directory>/<content hash>/<filename>`.

    Files are stored by content hash, so sending the same bytes again reuses the
    stored file (and, for a .miv, its cached preview). The partial file is moved
    into place only once every byte has arrived, so an interrupted upload is
    never mistaken for a complete one.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    handle, partial_name = tempfile.mkstemp(prefix=".upload-", dir=directory)
    partial = Path(partial_name)
    digest = hashlib.sha1()
    try:
        with os.fdopen(handle, "wb") as out:
            remaining = length
            while remaining > 0:
                chunk = stream.read(min(remaining, CHUNK_BYTES))
                if not chunk:
                    raise ValueError(f"the upload ended after {length - remaining} of "
                                     f"{length} bytes")
                digest.update(chunk)
                out.write(chunk)
                remaining -= len(chunk)
        target = directory / digest.hexdigest()[:16] / Path(filename).name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


class Library:
    """The .miv files on offer: files named on the command line, *.miv in named folders, uploads.

    Clips the web UI converted from ordinary videos are offered the same way,
    out of the folder `ConversionStore` writes them to.

    Rescanned on every request, so new runs show up without a restart.
    """

    def __init__(
        self,
        roots: Iterable[str | Path],
        upload_dir: str | Path,
        converted_dir: str | Path | None = None,
    ):
        """Offer `roots` (files or folders), uploads, and the clips converted here."""
        self.roots = [Path(root).expanduser().resolve() for root in roots]
        self.upload_dir = Path(upload_dir)
        self.converted_dir = Path(converted_dir) if converted_dir else None

    def _managed(self) -> list[tuple[Path, str]]:
        """The folders this server writes into, as `<folder>/<hash or id>/<name>.miv`."""
        folders = [(self.upload_dir, "upload")]
        if self.converted_dir is not None:
            folders.append((self.converted_dir, "converted"))
        return folders

    def sources(self) -> list[MivSource]:
        """Every source currently on disk, newest first."""
        paths: dict[Path, str] = {}
        for root in self.roots:
            for path in root.glob("*.miv") if root.is_dir() else [root]:
                if path.suffix.lower() == ".miv" and path.is_file():
                    paths[path.resolve()] = "file"
        for folder, origin in self._managed():
            for path in folder.glob("*/*.miv"):
                paths.setdefault(path.resolve(), origin)

        sources = []
        for path, origin in paths.items():
            try:
                sources.append(MivSource.from_path(path, origin=origin))
            except OSError:  # deleted while we were scanning
                continue
        return sorted(sources, key=lambda source: source.mtime_ns, reverse=True)

    def get(self, source_id: str) -> MivSource | None:
        """The source with this id, if it is still on disk and unchanged."""
        return next((s for s in self.sources() if s.source_id == source_id), None)

    def add_upload(self, filename: str, stream: BinaryIO, length: int) -> MivSource:
        """Store `length` bytes of an uploaded .miv from `stream` and return it as a source."""
        name = Path(filename).name
        if Path(name).suffix.lower() != ".miv":
            raise ValueError(f"expected a .miv file, got {name!r}")
        return MivSource.from_path(store_upload(self.upload_dir, name, stream, length),
                                   origin="upload")


# ---------------------------------------------------------------- mosaics


def tile_size(width: int, height: int, max_mosaic: tuple[int, int] = MAX_MOSAIC_SIZE
              ) -> tuple[int, int]:
    """One mosaic tile: the view size, scaled down so 3x3 tiles fit `max_mosaic`, kept even."""
    scale = min(1.0, max_mosaic[0] / (3 * width), max_mosaic[1] / (3 * height))
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def compose_mosaic(views: Sequence[np.ndarray], tile: tuple[int, int]) -> np.ndarray:
    """Place 9 [H,W,3] uint8 images, indexed by view, into a 3x3 mosaic in `MOSAIC_ORDER`."""
    tile_w, tile_h = tile
    mosaic = np.empty((3 * tile_h, 3 * tile_w, 3), dtype=np.uint8)
    for cell, view in enumerate(MOSAIC_ORDER):
        image = views[view]
        if image.shape[:2] != (tile_h, tile_w):
            image = np.asarray(Image.fromarray(image).resize((tile_w, tile_h), Image.BILINEAR))
        row, col = divmod(cell, 3)
        mosaic[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = image
    return mosaic


def colorize_depth(depth: np.ndarray, mask: np.ndarray, near: float, far: float) -> np.ndarray:
    """Turbo-colour inverse depth on a fixed [near, far] range, like the photo page.

    One range for the whole clip, rather than one per frame, keeps the colours
    from flickering. Invalid pixels get the photo page's flat dark colour.
    """
    valid = (mask != 0) & (depth > 0)
    low, high = 1.0 / far, 1.0 / near
    disparity = 1.0 / np.where(valid, depth, far)
    colored = turbo(((disparity - low) / max(high - low, 1e-12)).astype(np.float32))
    colored[~valid] = INVALID_DEPTH_COLOR
    return colored


def content_depth_range(decoded) -> tuple[float, float]:
    """The clip-wide depth range, from TMIV's per-view, per-frame `Depth_range`.

    TMIV fits each view's range to its content (dynamic depth range), so their
    union is the depth actually present in the clip.
    """
    ranges = [
        camera["Depth_range"]
        for frame in decoded.camera_update_frames
        for camera in decoded.config_at(frame)["cameras"]
    ]
    return float(min(r[0] for r in ranges)), float(max(r[1] for r in ranges))


class Mp4Writer:
    """Streams RGB frames into an H.264 MP4 tagged BT.709 limited range, as browsers expect."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        """Open `path` for `width` x `height` frames at `fps`."""
        self._container = av.open(str(path), "w", format="mp4",
                                  options={"movflags": "+faststart"})
        stream = self._container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        codec = stream.codec_context
        codec.color_primaries = codec.color_trc = codec.colorspace = 1  # BT.709
        codec.color_range = 1  # limited ("tv") range
        stream.options = {"crf": str(H264_CRF), "preset": "veryfast", "g": str(GOP_FRAMES)}
        self._stream = stream
        self._frames = 0

    def write(self, rgb: np.ndarray) -> None:
        """Append one [H,W,3] uint8 frame."""
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(
            format="yuv420p", dst_colorspace="ITU709", dst_color_range="MPEG")
        frame.pts = self._frames
        self._frames += 1
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        """Flush the encoder and finish the file."""
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def write_mosaics(decoded, directory: Path, on_frame: Callable[[int], None] | None = None
                  ) -> dict[str, Any]:
    """Write `color.mp4` and `depth.mp4` for a `DecodedMiv`; returns the layout metadata."""
    width, height = decoded.resolution
    tile = tile_size(width, height)
    near, far = content_depth_range(decoded)
    parts = {layer: directory / f"{layer}.part.mp4" for layer in LAYERS}
    writers = {layer: Mp4Writer(path, 3 * tile[0], 3 * tile[1], decoded.fps)
               for layer, path in parts.items()}
    frames = 0
    try:
        for views in decoded.frames():
            ordered = [views[name] for name in decoded.view_names]  # v0..v8
            writers["color"].write(compose_mosaic([v.rgb for v in ordered], tile))
            writers["depth"].write(compose_mosaic(
                [colorize_depth(v.depth, v.mask, near, far) for v in ordered], tile))
            frames += 1
            if on_frame is not None:
                on_frame(frames)
    finally:
        for writer in writers.values():
            writer.close()
    for layer, path in parts.items():
        os.replace(path, directory / f"{layer}.mp4")

    return {
        "frame_count": frames,
        "fps": decoded.fps,
        "view_size": [width, height],
        "tile_size": list(tile),
        "mosaic_size": [3 * tile[0], 3 * tile[1]],
        "mosaic_order": list(MOSAIC_ORDER),
        "depth_range_m": [near, far],
        "camera_update_frames": list(decoded.camera_update_frames),
    }


# ---------------------------------------------------------------- previews


@dataclass
class Preview:
    """The browser preview of one source: its build progress, then its files."""

    source: MivSource
    directory: Path
    state: str = "queued"  # queued | running | done | error
    stage: str = "queued"  # queued | probing | decoding | encoding | done | error
    progress: float = 0.0
    frame_count: int | None = None
    frames_done: int = 0
    error: str | None = None
    meta: dict | None = None
    decoded_dir: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_stage(self, stage: str, progress: float, frames_done: int = 0) -> None:
        """Record build progress."""
        with self._lock:
            self.state, self.stage = "running", stage
            self.progress, self.frames_done = progress, frames_done

    def start_decoding(self, frame_count: int, decoded_dir: Path) -> None:
        """Enter the decode stage; its progress is read off TmivDecoder's output files."""
        with self._lock:
            self.state, self.stage, self.progress = "running", "decoding", 0.0
            self.frame_count, self.frames_done, self.decoded_dir = frame_count, 0, decoded_dir

    def finish(self, meta: dict) -> None:
        """Mark the preview ready."""
        with self._lock:
            self.state = self.stage = "done"
            self.progress, self.meta, self.decoded_dir = 1.0, meta, None
            self.frame_count = self.frames_done = meta["frame_count"]

    def fail(self, message: str) -> None:
        """Mark the build failed."""
        with self._lock:
            self.state = self.stage = "error"
            self.error, self.decoded_dir = message, None

    def status(self, include_meta: bool = True) -> dict[str, Any]:
        """A JSON-serializable snapshot for the client to poll."""
        with self._lock:
            if self.stage == "decoding" and self.decoded_dir is not None and self.frame_count:
                try:
                    written = decoded_frames_written(self.decoded_dir)
                except OSError:  # a file appeared or vanished mid-scan; keep the last count
                    written = self.frames_done
                self.frames_done = min(self.frame_count, written)
                self.progress = DECODE_SHARE * self.frames_done / self.frame_count
            status = {
                "source_id": self.source.source_id,
                "state": self.state,
                "stage": self.stage,
                "progress": round(self.progress, 3),
                "frame_count": self.frame_count,
                "frames_done": self.frames_done,
                "error": self.error,
            }
            if include_meta and self.state == "done":
                status["meta"] = self.meta
            return status

    def file(self, name: str) -> Path | None:
        """Path of a finished preview file (`color.mp4`, `depth.mp4`), else None."""
        if self.state != "done" or name not in PREVIEW_FILES:
            return None
        return self.directory / name


def load_cached_meta(directory: Path) -> dict | None:
    """The metadata of a complete preview cached in `directory`, else None."""
    meta = _read_json(directory / "meta.json")
    if meta is None or meta.get("version") != PREVIEW_VERSION:
        return None
    if not all((directory / name).is_file() for name in PREVIEW_FILES):
        return None
    return meta


def _timestamps(manifest: dict | None, frame_count: int, fps: float) -> list[float]:
    """Source timestamps from the manifest; MIV itself carries only a frame rate."""
    frames = (manifest or {}).get("frames") or []
    if len(frames) == frame_count and all("timestamp" in f for f in frames):
        return [float(f["timestamp"]) for f in frames]
    return [index / fps for index in range(frame_count)]


def _manifest_summary(manifest: dict | None) -> dict | None:
    if not manifest:
        return None
    keys = ("format", "qp", "intra_period", "bitrate_bps", "depth_range", "cfr")
    return {key: manifest.get(key) for key in keys}


def _report_summary(report: dict | None) -> dict | None:
    if not report:
        return None
    source = report.get("input") or {}
    stage1 = report.get("stage1") or {}
    return {
        "source_video": Path(str(source.get("path") or "")).name or None,
        "source_resolution": source.get("resolution"),
        "frames_processed": stage1.get("frames_processed"),
        "stage1_s_per_frame": (stage1.get("stage1_s_per_frame") or {}).get("mean"),
        "total_processing_s": (report.get("overall") or {}).get("total_processing_time_s"),
    }


def describe_error(exc: BaseException) -> str:
    """The error, plus the end of the tool's log for a TMIV failure (the log itself is removed)."""
    message = f"{type(exc).__name__}: {exc}"
    log_path = getattr(exc, "log_path", None)
    if log_path is not None:
        try:
            tail = Path(log_path).read_text(errors="replace").strip().splitlines()
        except OSError:
            tail = []
        if tail:
            message += "\n\nEnd of the log:\n" + "\n".join(tail[-LOG_TAIL_LINES:])
    return message


def build_preview(
    preview: Preview,
    tmiv: TmivInstall | None = None,
    decode: Callable[..., Any] = decode_miv,
    probe: Callable[..., int] = probe_frame_count,
    runner: Callable[..., None] = run_logged,
) -> None:
    """Decode `preview.source` and write its mosaic videos and meta.json.

    Never raises: a failure is recorded on the preview. `decode`, `probe` and
    `runner` are injectable so this runs in tests without TMIV.
    """
    source = preview.source
    work = preview.directory / "work"
    try:
        shutil.rmtree(preview.directory, ignore_errors=True)
        work.mkdir(parents=True)
        manifest = source.manifest()

        started = time.perf_counter()
        if manifest and manifest.get("frame_count"):
            frame_count = int(manifest["frame_count"])
        else:
            preview.set_stage("probing", 0.01)
            frame_count = probe(source.path, work, tmiv=tmiv, runner=runner)
        preview.start_decoding(frame_count, work / "decoded")
        decoded = decode(source.path, frame_count, work, tmiv=tmiv, runner=runner)
        decode_s = time.perf_counter() - started

        started = time.perf_counter()
        preview.set_stage("encoding", DECODE_SHARE)

        def on_frame(done: int) -> None:
            preview.set_stage("encoding", DECODE_SHARE + (1 - DECODE_SHARE) * done / frame_count,
                              frames_done=done)

        meta = write_mosaics(decoded, preview.directory, on_frame)
        meta.update(
            version=PREVIEW_VERSION,
            source=source.describe(),
            timestamps=_timestamps(manifest, meta["frame_count"], meta["fps"]),
            manifest=_manifest_summary(manifest),
            report=_report_summary(source.report()),
            timings={"decode_s": round(decode_s, 2),
                     "encode_s": round(time.perf_counter() - started, 2)},
        )
        (preview.directory / "meta.json").write_text(json.dumps(meta, indent=2))
        preview.finish(meta)
    except BaseException as exc:  # noqa: BLE001 - surfaced to the client verbatim
        preview.fail(describe_error(exc))
    finally:
        shutil.rmtree(work, ignore_errors=True)


class PreviewStore:
    """Previews by source id: built on worker threads one at a time, cached on disk."""

    def __init__(
        self,
        cache_dir: str | Path,
        tmiv: TmivInstall | None,
        tmiv_error: str | None = None,
        build: Callable[..., None] = build_preview,
    ):
        """Cache previews under `cache_dir`; decode with `tmiv` (None: say why via `tmiv_error`)."""
        self.cache_dir = Path(cache_dir)
        self.tmiv = tmiv
        self.tmiv_error = tmiv_error
        self._build = build
        self._previews: dict[str, Preview] = {}
        self._lock = threading.Lock()
        #: TmivDecoder and x264 each use every core; building two at once gains nothing.
        self._build_lock = threading.Lock()

    def _directory(self, source: MivSource) -> Path:
        return self.cache_dir / "previews" / source.source_id

    def get(self, source: MivSource) -> Preview | None:
        """The source's preview: in progress, finished, or cached on disk by an earlier run."""
        with self._lock:
            preview = self._previews.get(source.source_id)
            if preview is None:
                meta = load_cached_meta(self._directory(source))
                if meta is not None:
                    preview = Preview(source, self._directory(source))
                    preview.finish(meta)
                    self._previews[source.source_id] = preview
            return preview

    def ensure(self, source: MivSource) -> Preview:
        """Start building the source's preview unless it is built or building.

        A preview whose build failed is rebuilt, so fixing the cause and retrying works.
        """
        self.get(source)  # adopt a preview cached on disk
        with self._lock:
            preview = self._previews.get(source.source_id)
            if preview is not None and preview.state != "error":
                return preview
            preview = Preview(source, self._directory(source))
            self._previews[source.source_id] = preview
        threading.Thread(target=self._run, args=(preview,), daemon=True,
                         name=f"miv-preview-{source.source_id}").start()
        return preview

    def _run(self, preview: Preview) -> None:
        with self._build_lock:
            if self.tmiv is None:
                preview.fail(self.tmiv_error or "No TMIV build found.")
                return
            self._build(preview, tmiv=self.tmiv)
