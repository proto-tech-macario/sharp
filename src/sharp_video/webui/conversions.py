"""Turning an ordinary video into MIV from the browser (the Stage 2 pipeline).

The video page takes .mp4 (and the other containers FFmpeg reads) as well as
.miv: the upload is stored, `run_video_to_miv` -- the same pipeline as
`sharp_video_to_miv` -- runs it on a worker thread, and the MIV file it writes
joins the library, where it is previewed like any other .miv.

Stage 1 is the slow part (seconds to minutes per frame, on the GPU), so:

- conversions run one at a time, and queue behind each other;
- progress is read off the frame cache the pipeline writes as it goes, rather
  than by changing the pipeline to report it;
- a conversion can be stopped between frames, by raising out of the
  spatializer the pipeline calls per frame.

A conversion of a given video at given Stage 1 settings always uses the same
work folder, and the pipeline runs with `resume=True`. So a run that is
interrupted -- killed, rebooted, stopped by hand -- picks up from the last
finished frame when it is started again, instead of repeating hours of work.
The work folder is therefore kept unless the run succeeded.

Conversions stream: each spatial frame goes straight to the MIV encoder rather
than through a temporal HDF5 of the whole clip first, which for a long video
saves writing and reading back tens of gigabytes.

The MIV file, its manifest and its report are kept -- they are what the preview
and the info card read.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from ..miv.tmiv import TmivInstall
from ..pipeline import PipelineOptions, default_output_size, run_video_to_miv
from ..scheduler import FrameSelection
from ..spatializer import DEFAULT_ANGLE_DEG, CameraConfig, Stage1Spatializer
from ..video_io import probe_video
from .previews import MivSource, describe_error, store_upload

#: Containers offered to the file picker. The codec inside is what actually
#: decides (see `video_io.SUPPORTED_CODECS`); probing reports an unusable one.
VIDEO_SUFFIXES = (".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".ts", ".m2ts", ".mts",
                  ".mpg", ".mpeg", ".3gp")

#: Share of the progress bar owned by Stage 1; the MIV encode fills the rest.
STAGE1_SHARE = 0.85

#: View heights offered in the browser; `None` keeps the source size. Stage 1
#: cost grows with the view size, so the default is well below most sources.
VIEW_HEIGHTS = (None, 1080, 720, 540, 360)
DEFAULT_VIEW_HEIGHT = 720

#: Frames converted when the browser does not say: all of them. A long video is
#: a long conversion, so the page shows the count it is in for and can stop it.
DEFAULT_MAX_FRAMES = None

#: Conversions remembered (for the library list) after they finish.
MAX_CONVERSIONS = 12

QP_RANGE = (0, 51)

#: The camera angle, on the same terms as the photo page
#: (`sharp_spatialize.webui.jobs`): it is the same Stage 1 knob. Angles below
#: the slider's minimum are still accepted, they are just not offered.
MAX_ANGLE_DEG = 30.0
SLIDER_MIN_ANGLE_DEG = 2.0


class ConversionCancelled(BaseException):
    """Raised inside the pipeline to stop a conversion between frames.

    A `BaseException`, not an `Exception`: `runner.process_frames` journals a
    failing *frame* and carries on to find every failure in one pass, which is
    right for a broken frame and wrong for a run the user stopped.
    """


def spatialized_frames(work_dir: str | Path) -> int:
    """How many frames Stage 1 has finished, from the cache the runner writes (for progress)."""
    return sum(1 for _ in (Path(work_dir) / "stage1" / "frames").glob("*.h5"))


def _int(query: Mapping[str, list[str]], key: str, absent: int | None,
         blank: int | None) -> int | None:
    """One integer query parameter: `absent` if the key is missing, `blank` if it is empty.

    The two differ: a browser that never sends `max_frames` wants the default,
    while one that sends an empty Frames box is asking for the whole video.
    """
    if key not in query:
        return absent
    raw = (query.get(key) or [""])[0].strip()
    if not raw:
        return blank
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{key} must be a whole number, got {raw!r}") from None


@dataclass(frozen=True)
class ConversionOptions:
    """What the browser may choose about a conversion; the rest comes from the server."""

    start_frame: int = 0
    max_frames: int | None = DEFAULT_MAX_FRAMES  # None: every frame to the end
    view_height: int | None = DEFAULT_VIEW_HEIGHT  # None: the source height
    angle_deg: float = DEFAULT_ANGLE_DEG
    qp_texture: int = 22
    qp_geometry: int = 8
    intra_period: int = 32
    threads: int = 4

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError(f"the start frame must be 0 or more, got {self.start_frame}")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError(f"the frame count must be 1 or more, got {self.max_frames}")
        if self.view_height is not None and self.view_height < 64:
            raise ValueError(f"the view height must be at least 64, got {self.view_height}")
        if not 0 < self.angle_deg <= MAX_ANGLE_DEG:
            raise ValueError(f"the camera angle must be more than 0 and at most "
                             f"{MAX_ANGLE_DEG} degrees, got {self.angle_deg}")
        for name in ("qp_texture", "qp_geometry"):
            value = getattr(self, name)
            if not QP_RANGE[0] <= value <= QP_RANGE[1]:
                raise ValueError(f"{name} must be {QP_RANGE[0]}-{QP_RANGE[1]}, got {value}")
        if self.intra_period not in (16, 32):
            raise ValueError(f"the intra period must be 16 or 32, got {self.intra_period}")

    @classmethod
    def from_query(cls, query: Mapping[str, list[str]]) -> ConversionOptions:
        """Read the options off an upload's query string; raises ValueError on a bad one."""
        defaults = cls()
        # An empty (or zero) frame count converts the whole video, and an empty
        # view height keeps the source size; both are "no limit", not a number.
        height = _int(query, "view_height", defaults.view_height, None)
        angle = (query.get("angle_deg") or [""])[0].strip()
        try:
            angle_deg = float(angle) if angle else defaults.angle_deg
        except ValueError:
            raise ValueError(f"angle_deg must be a number, got {angle!r}") from None
        return cls(
            start_frame=_int(query, "start_frame", defaults.start_frame, 0) or 0,
            max_frames=_int(query, "max_frames", defaults.max_frames, None),
            view_height=height if height else None,
            angle_deg=angle_deg,
            qp_texture=_int(query, "qp_texture", defaults.qp_texture, defaults.qp_texture),
            qp_geometry=_int(query, "qp_geometry", defaults.qp_geometry, defaults.qp_geometry),
            intra_period=_int(query, "intra_period", defaults.intra_period,
                              defaults.intra_period),
        )

    def selection(self) -> FrameSelection:
        """These options as the pipeline's frame selection."""
        return FrameSelection(start_frame=self.start_frame, max_frames=self.max_frames).validate()

    def output_size(self, width: int, height: int) -> tuple[int, int]:
        """The view size for a source of `width` x `height`: scaled down, then TMIV-aligned."""
        if self.view_height is not None and height > self.view_height:
            width = round(width * self.view_height / height)
            height = self.view_height
        return default_output_size(width, height)

    def planned_frames(self, total: int) -> int:
        """How many of a `total`-frame source this selection converts."""
        available = max(0, total - self.start_frame)
        return min(available, self.max_frames) if self.max_frames is not None else available

    def frame_key(self) -> str:
        """Identifies the Stage 1 frames these options produce.

        Only the settings that change a spatialized frame count: the frame
        range picks which frames, and the QPs belong to the MIV encoder, so
        neither may make a resumed run reuse a frame it should not.
        """
        recipe = f"{self.view_height}\0{self.angle_deg}"
        return hashlib.sha1(recipe.encode()).hexdigest()[:8]

    def describe(self) -> dict[str, Any]:
        """These options as JSON, for the browser to show back."""
        return {
            "start_frame": self.start_frame,
            "max_frames": self.max_frames,
            "view_height": self.view_height,
            "angle_deg": self.angle_deg,
            "qp_texture": self.qp_texture,
            "qp_geometry": self.qp_geometry,
            "intra_period": self.intra_period,
        }


@dataclass
class Conversion:
    """One video-to-MIV run: its progress while it runs, then the .miv it produced."""

    conversion_id: str
    video_path: Path
    output_path: Path
    work_dir: Path
    options: ConversionOptions
    state: str = "queued"  # queued | running | done | error | cancelled
    stage: str = "queued"  # queued | reading | spatializing | encoding_miv | done | ...
    progress: float = 0.0
    frame_count: int | None = None  # frames this run converts, not the source's length
    frames_done: int = 0
    error: str | None = None
    video: dict | None = None  # the source video's VideoInfo
    resumed_from: int = 0  # frames an interrupted earlier run had already done
    miv: dict | None = None  # the MIV file's headline numbers, once written
    source_id: str | None = None  # the library id of that file, to preview it
    started_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0
    cancelling: bool = False
    _started: float = field(default_factory=time.perf_counter, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def name(self) -> str:
        """The source video's file name."""
        return self.video_path.name

    def set_stage(self, stage: str, progress: float) -> None:
        """Record which stage the run is in."""
        with self._lock:
            self.state, self.stage, self.progress = "running", stage, progress

    def start_spatializing(self, frame_count: int, video: dict, resumed_from: int = 0) -> None:
        """Enter Stage 1; its progress is counted off the frame cache."""
        with self._lock:
            self.state, self.stage, self.progress = "running", "spatializing", 0.0
            self.frame_count, self.video, self.resumed_from = frame_count, video, resumed_from
            self.frames_done = min(resumed_from, frame_count)

    def finish(self, result, source: MivSource) -> None:
        """Mark the run done; `source` is the MIV file as the library will offer it."""
        with self._lock:
            self.state = self.stage = "done"
            self.progress, self.source_id = 1.0, source.source_id
            self.frames_done = self.frame_count = result.frame_count
            self.elapsed_s = time.perf_counter() - self._started
            self.miv = {
                "name": source.name,
                "frame_count": result.frame_count,
                "fps": result.fps,
                "size_bytes": result.size_bytes,
                "bitrate_bps": result.bitrate_bps,
                "cfr": result.cfr,
            }

    def fail(self, message: str, state: str = "error") -> None:
        """Mark the run failed (or, with `state="cancelled"`, stopped on request)."""
        with self._lock:
            self.state = self.stage = state
            self.error = message
            self.elapsed_s = time.perf_counter() - self._started

    def cancel(self) -> None:
        """Ask the run to stop after the frame it is on."""
        with self._lock:
            self.cancelling = True

    def guard(self, spatialize: Callable) -> Callable:
        """Wrap the per-frame Stage 1 call so a cancelled run stops before the next frame."""

        def guarded(frame):
            if self.cancelling:
                raise ConversionCancelled(f"{self.name}: stopped after {self.frames_done} frames")
            return spatialize(frame)

        return guarded

    def status(self) -> dict[str, Any]:
        """A JSON-serializable snapshot for the browser to poll."""
        with self._lock:
            if self.stage == "spatializing" and self.frame_count:
                try:
                    done = spatialized_frames(self.work_dir)
                except OSError:  # a file appeared or vanished mid-scan; keep the last count
                    done = self.frames_done
                self.frames_done = min(self.frame_count, done)
                self.progress = STAGE1_SHARE * self.frames_done / self.frame_count
                if self.frames_done == self.frame_count:
                    self.stage, self.progress = "encoding_miv", STAGE1_SHARE
            return {
                "id": self.conversion_id,
                "name": self.name,
                "state": self.state,
                "stage": self.stage,
                "progress": round(self.progress, 3),
                "frame_count": self.frame_count,
                "frames_done": self.frames_done,
                "cancelling": self.cancelling,
                "resumed_from": self.resumed_from,
                "error": self.error,
                "video": self.video,
                "miv": self.miv,
                "source_id": self.source_id,
                "options": self.options.describe(),
                "started_at": self.started_at,
                "elapsed_s": round(self.elapsed_s or (time.perf_counter() - self._started), 1),
            }


def run_conversion(
    conversion: Conversion,
    tmiv: TmivInstall | None = None,
    checkpoint_path: Path | None = None,
    device: str = "default",
    precision: str = "fp16",
    probe: Callable[..., Any] = probe_video,
    run: Callable[..., Any] = run_video_to_miv,
    spatializer: Callable[..., Callable] = Stage1Spatializer,
) -> None:
    """Convert `conversion.video_path` into MIV. Never raises: failures land on `conversion`.

    `probe`, `run` and `spatializer` are injectable, so this runs in tests
    without a GPU or TMIV.
    """
    try:
        conversion.set_stage("reading", 0.01)
        video = probe(conversion.video_path)
        options = conversion.options
        planned = options.planned_frames(video.frame_count)
        if planned <= 0:
            raise ValueError(
                f"{conversion.name} has {video.frame_count} frames, so there is nothing to "
                f"convert from frame {options.start_frame} on")
        conversion.start_spatializing(planned, video.as_dict(),
                                      resumed_from=spatialized_frames(conversion.work_dir))

        width, height = options.output_size(video.width, video.height)
        spatialize = conversion.guard(spatializer(
            conversion.work_dir / "stage1", camera=CameraConfig(angle_deg=options.angle_deg),
            output_size=(width, height), checkpoint_path=checkpoint_path, device=device,
            precision=precision,
        ))
        conversion.output_path.parent.mkdir(parents=True, exist_ok=True)
        result = run(
            PipelineOptions(
                input=conversion.video_path, output=conversion.output_path,
                # Streaming: no temporal HDF5 of the whole clip, so a long video
                # is not written out and read back in full before encoding.
                mode="streaming", resume=True,
                work_dir=conversion.work_dir, selection=options.selection(),
                camera=CameraConfig(angle_deg=options.angle_deg), output_size=(width, height),
                checkpoint=checkpoint_path, device=device, precision=precision,
                qp_texture=options.qp_texture, qp_geometry=options.qp_geometry,
                intra_period=options.intra_period, threads=options.threads,
                # TMIV's raw views and atlases are ~150 MB a frame at 1280x720;
                # nothing here reads them back, so they go as soon as TMIV is done.
                keep_intermediate=False,
            ),
            spatialize=spatialize,
            tmiv=tmiv,
        )
        conversion.finish(result.miv, MivSource.from_path(result.miv.path, origin="converted"))
    except ConversionCancelled as stopped:
        shutil.rmtree(conversion.output_path.parent, ignore_errors=True)
        conversion.fail(str(stopped), state="cancelled")
    except BaseException as exc:  # noqa: BLE001 - surfaced to the browser verbatim
        shutil.rmtree(conversion.output_path.parent, ignore_errors=True)
        conversion.fail(describe_error(exc))
    else:
        # Only a finished run may drop its frame cache: it is what a stopped or
        # crashed one resumes from, and it costs seconds per frame to rebuild.
        shutil.rmtree(conversion.work_dir, ignore_errors=True)


class ConversionStore:
    """Video uploads and their conversions: one at a time, newest first."""

    def __init__(
        self,
        cache_dir: str | Path,
        tmiv: TmivInstall | None,
        tmiv_error: str | None = None,
        checkpoint_path: Path | None = None,
        device: str = "default",
        precision: str = "fp16",
        run: Callable[..., None] = run_conversion,
    ):
        """Convert with `tmiv` (None: say why via `tmiv_error`) and Stage 1 on `device`."""
        self.cache_dir = Path(cache_dir)
        self.video_dir = self.cache_dir / "videos"
        self.converted_dir = self.cache_dir / "converted"
        self.tmiv = tmiv
        self.tmiv_error = tmiv_error
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.precision = precision
        self._run = run
        self._conversions: dict[str, Conversion] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        #: Stage 1 owns the GPU and TMIV owns the cores; two at once only thrash.
        self._run_lock = threading.Lock()

    def add_upload(self, filename: str, stream: BinaryIO, length: int) -> Path:
        """Store `length` bytes of an uploaded video from `stream` and return its path."""
        name = Path(filename).name
        if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"expected a video file ({', '.join(VIDEO_SUFFIXES)}), got {name!r}")
        return store_upload(self.video_dir, name, stream, length)

    def work_dir_for(self, video_path: Path, options: ConversionOptions) -> Path:
        """Where this video, at these Stage 1 settings, keeps its frames.

        Uploads are stored under a folder named for their content hash, so the
        same video converted the same way always lands here -- which is what
        lets an interrupted run be resumed simply by starting it again.
        """
        return self.cache_dir / "work" / f"{video_path.parent.name}-{options.frame_key()}"

    def start(self, video_path: str | Path, options: ConversionOptions) -> Conversion:
        """Queue a conversion of `video_path` and return it straight away.

        Frames left by an earlier interrupted run of the same video at the same
        settings are reused, so this resumes rather than starting over.
        """
        video_path = Path(video_path)
        conversion_id = uuid.uuid4().hex[:12]
        conversion = Conversion(
            conversion_id=conversion_id,
            video_path=video_path,
            output_path=self.converted_dir / conversion_id / f"{video_path.stem}.miv",
            work_dir=self.work_dir_for(video_path, options),
            options=options,
        )
        with self._lock:
            self._conversions[conversion_id] = conversion
            self._order.append(conversion_id)
            self._forget_old()
        threading.Thread(target=self._work, args=(conversion,), daemon=True,
                         name=f"miv-convert-{conversion_id}").start()
        return conversion

    def _forget_old(self) -> None:
        """Drop the oldest finished conversions; called with the lock held."""
        while len(self._order) > MAX_CONVERSIONS:
            for position, conversion_id in enumerate(self._order):
                if self._conversions[conversion_id].state in ("done", "error", "cancelled"):
                    del self._conversions[self._order.pop(position)]
                    break
            else:  # every record is still running
                return

    def get(self, conversion_id: str) -> Conversion | None:
        """The conversion with this id, if it is still remembered."""
        with self._lock:
            return self._conversions.get(conversion_id)

    def recent(self) -> list[Conversion]:
        """Every remembered conversion, newest first."""
        with self._lock:
            return [self._conversions[cid] for cid in reversed(self._order)]

    def cancel(self, conversion_id: str) -> Conversion | None:
        """Ask a running conversion to stop; a queued one never starts."""
        conversion = self.get(conversion_id)
        if conversion is not None and conversion.state in ("queued", "running"):
            conversion.cancel()
        return conversion

    def _work(self, conversion: Conversion) -> None:
        with self._run_lock:
            if conversion.cancelling:  # cancelled while it waited its turn
                return conversion.fail(f"{conversion.name}: stopped before it started",
                                       state="cancelled")
            if self.tmiv is None:
                return conversion.fail(self.tmiv_error or "No TMIV build found.")
            self._run(conversion, tmiv=self.tmiv, checkpoint_path=self.checkpoint_path,
                      device=self.device, precision=self.precision)
