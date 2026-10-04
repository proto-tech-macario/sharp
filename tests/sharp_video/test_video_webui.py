"""Tests for sharp_video.webui: video-to-MIV conversion, MIV previews, the HTTP layer.

TmivDecoder is replaced by a fake serving synthetic decoded frames, and the
Stage 2 pipeline by a fake that writes a stand-in .miv, so the preview build
(mosaic layout, depth colours, H.264 files, caching), the conversion flow
(progress, cancellation, the converted clip joining the library) and the HTTP
layer all run without TMIV or a GPU. The slow test at the end uses the real
decoder.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
from sharp_spatialize.webui.jobs import INVALID_DEPTH_COLOR, JobStore, turbo
from sharp_video.miv.decode import (
    DecodedView,
    decoded_frames_written,
    frame_count_from_parser_dump,
)
from sharp_video.miv.tmiv import TmivError, TmivInstall
from sharp_video.miv.tmiv_config import VIEW_NAMES
from sharp_video.video_io import VideoInfo
from sharp_video.webui.conversions import (
    STAGE1_SHARE,
    Conversion,
    ConversionCancelled,
    ConversionOptions,
    ConversionStore,
    run_conversion,
    spatialized_frames,
)
from sharp_video.webui.previews import (
    MOSAIC_ORDER,
    Library,
    MivSource,
    Preview,
    PreviewStore,
    build_preview,
    colorize_depth,
    compose_mosaic,
    tile_size,
)
from sharp_video.webui.server import MediaServerConfig, SpatialMediaHandler
from synthetic import make_frames

POLL_TIMEOUT_S = 30.0

#: Seconds the faked pipeline spends per frame, standing in for Stage 1.
FAKE_FRAME_S = 0.2


class FakeDecoded:
    """Duck-types DecodedMiv over synthetic frames."""

    def __init__(self, frames, fps=30.0):
        """Serve `frames` as if TmivDecoder had reconstructed them."""
        self._frames = frames
        self.frame_count = len(frames)
        self.fps = fps
        height, width = frames[0].views[0].rgb.shape[:2]
        self.resolution = (width, height)
        self.view_names = list(VIEW_NAMES)
        self.camera_update_frames = list(range(len(frames)))

    def config_at(self, frame_index):
        """Per-view depth ranges fitted to the content, like TMIV's dynamic depth range."""
        cameras = []
        for view in self._frames[frame_index].views:
            valid = view.depth[view.mask == 1]
            near, far = (float(valid.min()), float(valid.max())) if valid.size else (1.0, 10.0)
            cameras.append({"Depth_range": [near, far]})
        return {"cameras": cameras}

    def frames(self):
        """Yield `{view_name: DecodedView}` per frame, like `DecodedMiv.frames()`."""
        for frame in self._frames:
            yield {
                name: DecodedView(rgb=v.rgb, depth=v.depth, mask=v.mask, K=v.K, R=v.R, C=v.C)
                for name, v in zip(VIEW_NAMES, frame.views)
            }


def fake_decoder(frames, calls=None):
    """A `decode_miv` stand-in that returns the first `frame_count` of `frames`."""

    def decode(miv_path, frame_count, work_dir, tmiv=None, runner=None):
        if calls is not None:
            calls.append(frame_count)
        return FakeDecoded(frames[:frame_count])

    return decode


def write_miv(folder: Path, name: str = "clip.miv", frame_count: int | None = 3) -> Path:
    """A stand-in .miv file, with a Stage 2 manifest unless `frame_count` is None."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"\x00MIV" * 64)
    if frame_count is not None:
        Path(f"{path}.json").write_text(json.dumps({
            "frame_count": frame_count, "fps": 30.0, "resolution": [32, 32],
            "qp": {"texture": 22, "geometry": 8},
            "frames": [{"timestamp": 2.0 + i / 30.0} for i in range(frame_count)],
        }))
    return path


def make_video_info(frame_count: int = 10, width: int = 1920, height: int = 1080) -> VideoInfo:
    """What `probe_video` would report about a source video."""
    return VideoInfo(
        path="clip.mp4", container="mov,mp4", codec="h264", width=width, height=height,
        fps=30.0, frame_count=frame_count, duration=frame_count / 30.0, time_base="1/30000",
        pix_fmt="yuv420p", color_primaries="bt709", color_trc="bt709", colorspace="bt709",
        color_range="tv",
    )


#: A `Stage1Spatializer` stand-in: no checkpoint, no GPU, no rendering.
fake_spatializer = lambda *args, **kwargs: (lambda frame: frame)  # noqa: E731


def cancelling_spatializer(conversion: Conversion, after: int):
    """A spatializer that cancels `conversion` once it has done `after` frames.

    That is how a stop really arrives: the guard around this call is what
    raises, on the frame after the one the user pressed Stop during.
    """

    def factory(*args, **kwargs):
        done = []

        def spatialize(frame):
            done.append(frame)
            if len(done) >= after:
                conversion.cancel()
            return frame

        return spatialize

    return factory


def fake_pipeline(source_frames: int = 0, fail: Exception | None = None, delay: float = 0.0):
    """A `run_video_to_miv` stand-in: writes a stand-in .miv, no GPU and no TMIV.

    `source_frames` frames are pushed through the injected spatializer first, as
    the real pipeline does, so the cancellation guard is exercised. `delay`
    stands in for the seconds a real Stage 1 frame takes, so a test can cancel a
    run while it is between frames.
    """

    def run(options, spatialize=None, tmiv=None, **kwargs):
        cache = Path(options.work_dir) / "stage1" / "frames"
        cache.mkdir(parents=True, exist_ok=True)
        for index in range(source_frames):
            if spatialize is not None and not (cache / f"{index:06d}.h5").exists():
                spatialize(index)  # a resumed frame is read back, not spatialized again
            (cache / f"{index:06d}.h5").write_bytes(b"frame")
            time.sleep(delay)
        if fail is not None:
            raise fail
        path = Path(options.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_miv(path.parent, path.name, frame_count=source_frames or 2)
        miv = SimpleNamespace(path=path, frame_count=source_frames or 2, fps=30.0,
                              size_bytes=path.stat().st_size, bitrate_bps=1234.0, cfr=True)
        return SimpleNamespace(miv=miv)

    return run


def make_conversion(tmp_path: Path, options: ConversionOptions | None = None) -> Conversion:
    """One conversion of a stand-in video, writing where the store would put it."""
    video = tmp_path / "videos" / "clip.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"not really an mp4")
    return Conversion(
        conversion_id="c0ffee", video_path=video,
        output_path=tmp_path / "converted" / "c0ffee" / "clip.miv",
        work_dir=tmp_path / "work" / "c0ffee", options=options or ConversionOptions(),
    )


def wait_for_conversion(conversion: Conversion) -> dict:
    """Poll a conversion until it finishes, fails or stops."""
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        status = conversion.status()
        if status["state"] in ("done", "error", "cancelled"):
            return status
        time.sleep(0.02)
    raise AssertionError(f"conversion did not settle within {POLL_TIMEOUT_S}s")


def read_video(path: Path) -> list[np.ndarray]:
    """Every frame of an MP4 as RGB arrays."""
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def wait_until_settled(preview: Preview) -> dict:
    """Poll a preview until its build finishes or fails."""
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        status = preview.status()
        if status["state"] in ("done", "error"):
            return status
        time.sleep(0.02)
    raise AssertionError(f"preview did not settle within {POLL_TIMEOUT_S}s")


@pytest.fixture(scope="module")
def frames():
    """Three drifting synthetic frames, 32x32 per view."""
    return make_frames(3, drift=0.3, motion=0.2, tilt=0.4)


# ------------------------------------------------------------------ mosaics


def test_tile_size_keeps_720p_views_and_scales_larger_ones_to_fit():
    """Tiles stay at view size up to 720p; bigger views shrink so the mosaic fits 3840x2160."""
    assert tile_size(1280, 720) == (1280, 720)
    assert tile_size(640, 480) == (640, 480)
    assert tile_size(1920, 1080) == (1280, 720)
    width, height = tile_size(1000, 1500)
    assert 3 * height <= 2160 and width % 2 == 0 and height % 2 == 0


def test_the_mosaic_puts_each_view_where_its_camera_is():
    """The top row holds the cameras above the reference view (v6..v8), v4 is central."""
    views = [np.full((4, 6, 3), 20 * index, dtype=np.uint8) for index in range(9)]
    mosaic = compose_mosaic(views, (6, 4))

    assert mosaic.shape == (12, 18, 3)
    for cell, view in enumerate(MOSAIC_ORDER):
        row, col = divmod(cell, 3)
        assert (mosaic[row * 4:(row + 1) * 4, col * 6:(col + 1) * 6] == 20 * view).all(), cell
    assert MOSAIC_ORDER[0] == 6 and MOSAIC_ORDER[4] == 4 and MOSAIC_ORDER[8] == 2


def test_the_mosaic_scales_views_to_the_tile_size():
    """Views larger than the tile are resized into it."""
    views = [np.full((8, 12, 3), 100, dtype=np.uint8) for _ in range(9)]
    assert compose_mosaic(views, (6, 4)).shape == (12, 18, 3)


def test_depth_colours_run_near_to_far_on_one_fixed_range():
    """Near is the hot end of turbo, far the cold end, invalid pixels are flat dark."""
    depth = np.array([[2.0, 10.0, 5.0, 3.0]], dtype=np.float32)
    mask = np.array([[1, 1, 0, 1]], dtype=np.uint8)
    colored = colorize_depth(depth, mask, near=2.0, far=10.0)

    assert (colored[0, 0] == turbo(np.array([1.0]))[0]).all()
    assert (colored[0, 1] == turbo(np.array([0.0]))[0]).all()
    assert (colored[0, 2] == INVALID_DEPTH_COLOR).all()


# ------------------------------------------------------- decoder helpers


def test_frame_count_comes_from_the_parser_dump():
    """One atlas tile header per atlas per frame: 6 headers over 2 atlases is 3 frames."""
    dump = "vps_atlas_count_minus1=1\n" + "ath_id=0\nath_type=I_TILE\n" * 6
    assert frame_count_from_parser_dump(dump) == 3

    with pytest.raises(ValueError):
        frame_count_from_parser_dump("not a parser dump")


def test_decoding_progress_counts_whole_frames_written(tmp_path):
    """Progress reads view 0's texture file: 2.5 frames of 4x2 10-bit 4:2:0 is 2 frames."""
    assert decoded_frames_written(tmp_path) == 0
    (tmp_path / "00_tex_4x2_yuv420p10le.yuv").write_bytes(b"\x00" * 60)  # 24 bytes a frame
    assert decoded_frames_written(tmp_path) == 2


# -------------------------------------------------------------- building


def test_a_preview_is_two_playable_mosaic_videos_and_its_metadata(tmp_path, frames):
    """Colour and depth mosaics hold every frame; meta.json describes the layout."""
    source = MivSource.from_path(write_miv(tmp_path / "clips"))
    preview = Preview(source, tmp_path / "preview")

    build_preview(preview, decode=fake_decoder(frames))

    status = preview.status()
    assert status["state"] == "done", status["error"]
    meta = status["meta"]
    assert meta["frame_count"] == 3 and meta["fps"] == 30.0
    assert meta["tile_size"] == [32, 32] and meta["mosaic_size"] == [96, 96]
    assert meta["mosaic_order"] == list(MOSAIC_ORDER)
    assert meta["timestamps"] == pytest.approx([2.0, 2.0 + 1 / 30, 2.0 + 2 / 30])
    assert meta["manifest"]["qp"] == {"texture": 22, "geometry": 8}
    near, far = meta["depth_range_m"]
    assert 0 < near < far

    color = read_video(preview.file("color.mp4"))
    depth = read_video(preview.file("depth.mp4"))
    assert len(color) == len(depth) == 3
    assert color[0].shape == depth[0].shape == (96, 96, 3)
    centre = color[0][32:64, 32:64].astype(int)
    assert np.abs(centre - frames[0].views[4].rgb).mean() < 20, "v4 should be the centre tile"

    assert json.loads((preview.directory / "meta.json").read_text()) == meta
    assert not (preview.directory / "work").exists(), "decoder output must be cleaned up"


def test_without_a_manifest_the_frame_count_is_probed(tmp_path, frames):
    """A bare .miv (say, uploaded alone) is counted with TmivParser before decoding."""
    source = MivSource.from_path(write_miv(tmp_path, frame_count=None))
    preview = Preview(source, tmp_path / "preview")
    calls = []

    build_preview(preview, decode=fake_decoder(frames, calls),
                  probe=lambda *args, **kwargs: 2)

    assert preview.state == "done", preview.error
    assert calls == [2]
    assert preview.meta["timestamps"] == pytest.approx([0.0, 1 / 30])
    assert preview.meta["manifest"] is None


def test_a_failed_decode_reports_the_end_of_the_tmiv_log(tmp_path):
    """TmivDecoder's log is deleted with the work dir, so its last lines go into the error."""
    source = MivSource.from_path(write_miv(tmp_path))
    preview = Preview(source, tmp_path / "preview")

    def failing_decode(miv_path, frame_count, work_dir, tmiv=None, runner=None):
        log = Path(work_dir) / "logs" / "decode.log"
        log.parent.mkdir(parents=True)
        log.write_text("starting\nERROR: bitstream is corrupt\n")
        raise TmivError(f"TmivDecoder failed (exit 1); see {log}", log)

    build_preview(preview, decode=failing_decode)

    assert preview.state == "error"
    assert "TmivDecoder failed" in preview.error
    assert "ERROR: bitstream is corrupt" in preview.error
    assert not (preview.directory / "work").exists()


def test_previews_are_built_once_and_then_served_from_the_disk_cache(tmp_path, frames):
    """A second store over the same cache finds the preview without decoding again."""
    source = MivSource.from_path(write_miv(tmp_path / "clips"))
    calls = []
    build = partial(build_preview, decode=fake_decoder(frames, calls))
    tmiv = TmivInstall(tmp_path / "tmiv")

    store = PreviewStore(tmp_path / "cache", tmiv, build=build)
    assert store.get(source) is None
    first = store.ensure(source)
    assert wait_until_settled(first)["state"] == "done"
    assert store.ensure(source) is first, "a finished preview is not rebuilt"

    again = PreviewStore(tmp_path / "cache", tmiv, build=build).get(source)
    assert again is not None and again.state == "done"
    assert again.meta == first.meta
    assert calls == [3]


def test_a_failed_preview_is_retried_on_the_next_request(tmp_path, frames):
    """Fix the cause, ask again, and the preview builds."""
    source = MivSource.from_path(write_miv(tmp_path / "clips"))
    outcomes = iter([RuntimeError("disk full"), None])

    def flaky_build(preview, tmiv=None):
        outcome = next(outcomes)
        if outcome is not None:
            return preview.fail(str(outcome))
        build_preview(preview, tmiv=tmiv, decode=fake_decoder(frames))

    store = PreviewStore(tmp_path / "cache", TmivInstall(tmp_path), build=flaky_build)
    assert wait_until_settled(store.ensure(source))["state"] == "error"
    assert wait_until_settled(store.ensure(source))["state"] == "done"


def test_without_tmiv_a_preview_fails_with_the_reason(tmp_path):
    """No TMIV build: the preview says why instead of hanging."""
    source = MivSource.from_path(write_miv(tmp_path))
    store = PreviewStore(tmp_path / "cache", None, "No TMIV build found. Build it with ...")

    status = wait_until_settled(store.ensure(source))

    assert status["state"] == "error"
    assert "No TMIV build found" in status["error"]


# --------------------------------------------------------------- library


def test_the_library_lists_folders_files_and_uploads(tmp_path):
    """*.miv in folders, explicitly named files, and uploads; nothing else."""
    in_folder = write_miv(tmp_path / "out", "a.miv")
    (tmp_path / "out" / "notes.txt").write_text("not a video")
    named = write_miv(tmp_path / "elsewhere", "b.miv")
    library = Library([tmp_path / "out", named], tmp_path / "uploads")
    with open(in_folder, "rb") as stream:
        uploaded = library.add_upload("c.miv", stream, in_folder.stat().st_size)

    sources = {source.name: source for source in library.sources()}

    assert set(sources) == {"a.miv", "b.miv", "c.miv"}
    assert sources["c.miv"].uploaded and not sources["a.miv"].uploaded
    assert library.get(uploaded.source_id) == uploaded
    assert library.get("feedfacecafe") is None


def test_a_changed_file_gets_a_new_id(tmp_path):
    """Re-encoding a clip in place must not show its stale preview."""
    path = write_miv(tmp_path)
    before = MivSource.from_path(path).source_id
    path.write_bytes(b"\x00MIV" * 128)
    assert MivSource.from_path(path).source_id != before


def test_uploads_are_stored_once_and_must_be_complete_miv_files(tmp_path):
    """The same bytes twice are one source; short or non-.miv uploads are refused."""
    library = Library([], tmp_path / "uploads")
    payload = b"\x00MIV" * 100

    first = library.add_upload("clip.miv", _Stream(payload), len(payload))
    second = library.add_upload("clip.miv", _Stream(payload), len(payload))
    assert first == second and first.path.read_bytes() == payload

    with pytest.raises(ValueError, match="ended after"):
        library.add_upload("short.miv", _Stream(payload[:10]), len(payload))
    with pytest.raises(ValueError, match=".miv"):
        library.add_upload("clip.h5", _Stream(payload), len(payload))
    assert [p.name for p in (tmp_path / "uploads").rglob("*") if p.is_file()] == ["clip.miv"]


# ------------------------------------------------------------- conversions


def test_conversion_options_come_off_the_query_string():
    """Every option the browser can set is read, and an empty box means the default."""
    options = ConversionOptions.from_query({
        "start_frame": ["12"], "max_frames": ["4"], "view_height": ["540"],
        "angle_deg": ["7.5"], "qp_texture": ["30"], "qp_geometry": ["12"],
    })
    assert options.start_frame == 12 and options.max_frames == 4
    assert options.view_height == 540 and options.angle_deg == 7.5
    assert (options.qp_texture, options.qp_geometry) == (30, 12)
    assert ConversionOptions.from_query({}) == ConversionOptions()


def test_the_whole_video_is_converted_unless_a_frame_count_is_asked_for():
    """There is no frame cap: an empty box, or no box at all, converts all of it."""
    for query in ({}, {"max_frames": [""]}):
        options = ConversionOptions.from_query(query)
        assert options.max_frames is None
        assert options.planned_frames(18_000) == 18_000  # ten minutes at 30 fps
        assert options.selection().max_frames is None
    assert ConversionOptions().max_frames is None


def test_a_zero_view_height_keeps_the_source_size():
    """"Source size" is "not set", not a number."""
    options = ConversionOptions.from_query({"view_height": ["0"]})
    assert options.view_height is None
    assert options.output_size(1920, 1080) == (1920, 1080)


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ({"max_frames": ["0"]}, "1 or more"),
        ({"start_frame": ["-1"]}, "0 or more"),
        ({"angle_deg": ["90"]}, "camera angle"),
        ({"angle_deg": ["0"]}, "camera angle"),
        ({"angle_deg": ["wide"]}, "must be a number"),
        ({"qp_texture": ["99"]}, "qp_texture"),
        ({"max_frames": ["lots"]}, "whole number"),
    ],
)
def test_impossible_conversion_options_are_refused_with_the_reason(query, message):
    """A bad option is caught before any video is uploaded, and says which one it was."""
    with pytest.raises(ValueError, match=message):
        ConversionOptions.from_query(query)


def test_views_are_scaled_down_to_the_chosen_height_and_kept_tmiv_aligned():
    """The view keeps its aspect, and both sides stay multiples of 8 (4:2:0, TMIV blocks)."""
    assert ConversionOptions(view_height=720).output_size(1920, 1080) == (1280, 720)
    assert ConversionOptions(view_height=720).output_size(1440, 1080) == (960, 720)
    # 540 is not a multiple of 8, so the aligned view is 536 -- as on the command line.
    assert ConversionOptions(view_height=540).output_size(1920, 1080) == (960, 536)
    assert ConversionOptions(view_height=720).output_size(640, 360) == (640, 360)


def test_only_the_selected_frames_are_counted():
    """The progress bar counts the frames this run converts, not the source's length."""
    assert ConversionOptions(start_frame=5, max_frames=10).planned_frames(100) == 10
    assert ConversionOptions(start_frame=95, max_frames=10).planned_frames(100) == 5
    assert ConversionOptions(start_frame=100, max_frames=10).planned_frames(100) == 0


def test_spatializing_progress_counts_the_frames_the_pipeline_has_cached(tmp_path):
    """Stage 1 progress is read off the runner's frame cache, without changing the pipeline."""
    conversion = make_conversion(tmp_path)
    conversion.start_spatializing(4, make_video_info().as_dict())
    cache = conversion.work_dir / "stage1" / "frames"
    cache.mkdir(parents=True)
    assert conversion.status()["progress"] == 0.0

    for index in range(2):
        (cache / f"{index:06d}.h5").write_bytes(b"frame")
    status = conversion.status()
    assert status["frames_done"] == 2
    assert status["progress"] == pytest.approx(STAGE1_SHARE / 2, abs=1e-3)
    assert status["stage"] == "spatializing"

    for index in range(2, 4):
        (cache / f"{index:06d}.h5").write_bytes(b"frame")
    assert conversion.status()["stage"] == "encoding_miv"  # Stage 1 done; TMIV has it now


def test_a_conversion_writes_a_miv_the_library_can_offer(tmp_path):
    """The happy path: the .miv lands where the library looks, with its id ready to preview."""
    conversion = make_conversion(tmp_path, ConversionOptions(max_frames=3))
    run_conversion(conversion, spatializer=fake_spatializer,
                   probe=lambda path: make_video_info(frame_count=10),
                   run=fake_pipeline(source_frames=3))

    status = conversion.status()
    assert status["state"] == "done" and status["error"] is None
    assert status["frame_count"] == 3 and status["miv"]["frame_count"] == 3
    assert status["video"]["frame_count"] == 10  # the source, not the selection

    library = Library([], tmp_path / "uploads", tmp_path / "converted")
    [source] = library.sources()
    assert source.origin == "converted" and source.source_id == status["source_id"]
    assert source.name == "clip.miv" and source.manifest() is not None


def test_the_bulky_intermediates_are_deleted_and_the_miv_is_kept(tmp_path):
    """The spatial sequence and frame cache dwarf the MIV file; only the MIV survives."""
    conversion = make_conversion(tmp_path)
    run_conversion(conversion, spatializer=fake_spatializer, probe=lambda path: make_video_info(),
                   run=fake_pipeline(source_frames=2))
    assert conversion.output_path.is_file()
    assert not conversion.work_dir.exists()


def test_a_failed_conversion_reports_why_and_leaves_no_half_written_clip(tmp_path):
    """A pipeline failure is surfaced verbatim, and the library is not offered a stub."""
    conversion = make_conversion(tmp_path)
    run_conversion(conversion, spatializer=fake_spatializer, probe=lambda path: make_video_info(),
                   run=fake_pipeline(source_frames=2,
                                     fail=RuntimeError("VVenC exited with 1")))

    status = conversion.status()
    assert status["state"] == "error" and "VVenC exited with 1" in status["error"]
    assert not conversion.output_path.parent.exists()
    assert Library([], tmp_path / "uploads", tmp_path / "converted").sources() == []
    # The frames it did finish are kept: they are what the next run resumes from.
    assert spatialized_frames(conversion.work_dir) == 2


def test_an_interrupted_run_keeps_its_frames_and_the_next_one_resumes(tmp_path):
    """Hours of Stage 1 survive a stop; starting again picks up where it left off."""
    options = ConversionOptions(max_frames=5)
    stopped = make_conversion(tmp_path, options)
    run_conversion(stopped, spatializer=cancelling_spatializer(stopped, after=3),
                   probe=lambda path: make_video_info(), run=fake_pipeline(source_frames=5))
    assert stopped.status()["state"] == "cancelled"
    assert spatialized_frames(stopped.work_dir) == 3  # kept, not swept up

    again = make_conversion(tmp_path, options)  # the same video, the same settings
    assert again.work_dir == stopped.work_dir
    run_conversion(again, spatializer=fake_spatializer, probe=lambda path: make_video_info(),
                   run=fake_pipeline(source_frames=5))
    assert again.status()["resumed_from"] == 3
    assert again.status()["state"] == "done"


def test_the_same_video_and_settings_always_use_one_work_folder(tmp_path):
    """Resuming works by starting again, so the folder cannot be per-run."""
    store = ConversionStore(tmp_path, None)
    payload = b"\x00\x00\x00 ftypisom" * 20
    video = store.add_upload("clip.mp4", _Stream(payload), len(payload))

    same = ConversionOptions(view_height=720, angle_deg=10.0, max_frames=5)
    also_same = ConversionOptions(view_height=720, angle_deg=10.0, max_frames=900, qp_texture=30)
    different = ConversionOptions(view_height=360, angle_deg=10.0)
    turned = ConversionOptions(view_height=720, angle_deg=14.0)

    # The frame range and the MIV quality do not change a spatialized frame.
    assert store.work_dir_for(video, same) == store.work_dir_for(video, also_same)
    # The view size and the camera angle do, so those frames must not be reused.
    assert store.work_dir_for(video, different) != store.work_dir_for(video, same)
    assert store.work_dir_for(video, turned) != store.work_dir_for(video, same)


def test_converting_past_the_end_of_a_video_says_so_instead_of_running(tmp_path):
    """Nothing is spatialized when the selection cannot select a frame."""
    conversion = make_conversion(tmp_path, ConversionOptions(start_frame=50))
    ran = []
    run_conversion(conversion, spatializer=fake_spatializer,
                   probe=lambda path: make_video_info(frame_count=10),
                   run=lambda *args, **kwargs: ran.append(1))
    assert ran == []
    assert "nothing to convert from frame 50" in conversion.status()["error"]


def test_a_cancelled_conversion_stops_at_the_next_frame(tmp_path):
    """Cancelling raises out of the per-frame Stage 1 call.

    The run stops there, rather than journalling a failed frame and carrying on.
    """
    conversion = make_conversion(tmp_path, ConversionOptions(max_frames=5))
    conversion.cancel()
    run_conversion(conversion, spatializer=fake_spatializer, probe=lambda path: make_video_info(),
                   run=fake_pipeline(source_frames=5))

    status = conversion.status()
    assert status["state"] == "cancelled" and status["cancelling"] is True
    assert not conversion.output_path.exists()


def test_the_guard_only_raises_once_cancelling(tmp_path):
    """The wrapper is transparent until the run is cancelled."""
    conversion = make_conversion(tmp_path)
    guarded = conversion.guard(lambda frame: f"spatialized {frame}")
    assert guarded(7) == "spatialized 7"
    conversion.cancel()
    with pytest.raises(ConversionCancelled):
        guarded(8)


def test_conversions_queue_and_are_remembered_newest_first(tmp_path):
    """Conversions run one at a time (Stage 1 owns the GPU) and stay listed afterwards."""
    store = ConversionStore(tmp_path, TmivInstall(tmp_path / "tmiv"),
                            run=partial(run_conversion, spatializer=fake_spatializer,
                                        probe=lambda p: make_video_info(),
                                        run=fake_pipeline(source_frames=1)))
    videos = []
    for name in ("a.mp4", "b.mp4"):
        payload = name.encode() * 40
        videos.append(store.add_upload(name, _Stream(payload), len(payload)))
    first, second = (store.start(path, ConversionOptions()) for path in videos)

    assert wait_for_conversion(first)["state"] == "done"
    assert wait_for_conversion(second)["state"] == "done"
    assert [c.name for c in store.recent()] == ["b.mp4", "a.mp4"]
    assert store.get(first.conversion_id) is first and store.get("nope") is None


def test_uploads_must_be_videos_and_are_stored_once(tmp_path):
    """The same video twice is one stored file; a .miv does not go down this path."""
    store = ConversionStore(tmp_path, None)
    payload = b"\x00\x00\x00 ftypisom" * 20
    first = store.add_upload("clip.mp4", _Stream(payload), len(payload))
    assert store.add_upload("clip.mp4", _Stream(payload), len(payload)) == first
    with pytest.raises(ValueError, match="expected a video file"):
        store.add_upload("clip.miv", _Stream(payload), len(payload))


def test_without_tmiv_a_conversion_fails_before_any_gpu_work(tmp_path):
    """The MIV encoder is unusable, so the run says so rather than spending hours first."""
    store = ConversionStore(tmp_path, None, "no .tmiv anywhere")
    payload = b"mp4" * 20
    conversion = store.start(store.add_upload("clip.mp4", _Stream(payload), len(payload)),
                             ConversionOptions())
    assert wait_for_conversion(conversion)["error"] == "no .tmiv anywhere"


class _Stream:
    """A file-like body that serves `data` in small reads."""

    def __init__(self, data: bytes):
        """Serve `data`."""
        self._data = data

    def read(self, size: int) -> bytes:
        """Return up to `size` bytes, like a socket file."""
        chunk, self._data = self._data[:min(size, 7)], self._data[min(size, 7):]
        return chunk


# ----------------------------------------------------------------- server


class Client:
    """Minimal HTTP client bound to one running test server."""

    def __init__(self, base_url: str) -> None:
        """Point the client at a running server's base URL."""
        self.base_url = base_url

    def request(self, path: str, method: str = "GET", body: bytes | None = None):
        """Return (status, headers, body_bytes); HTTP errors come back, not raised."""
        request = urllib.request.Request(self.base_url + path, data=body, method=method)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def json(self, path: str, method: str = "GET", body: bytes | None = None):
        """Same as `request`, with the body parsed as JSON."""
        status, _headers, payload = self.request(path, method=method, body=body)
        return status, json.loads(payload)

    def wait_for_preview(self, source_id: str) -> dict:
        """Poll a preview until it finishes or fails."""
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            _status, preview = self.json(f"/api/miv/previews/{source_id}")
            if preview.get("state") in ("done", "error"):
                return preview
            time.sleep(0.02)
        raise AssertionError(f"preview {source_id} did not settle within {POLL_TIMEOUT_S}s")


    def wait_for_conversion(self, conversion_id: str) -> dict:
        """Poll a conversion over HTTP until it finishes, fails or stops."""
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            _status, conversion = self.json(f"/api/video/conversions/{conversion_id}")
            if conversion.get("state") in ("done", "error", "cancelled"):
                return conversion
            time.sleep(0.02)
        raise AssertionError(f"conversion {conversion_id} did not settle in {POLL_TIMEOUT_S}s")


@pytest.fixture
def served(tmp_path, frames):
    """A running web UI over a folder with one .miv, with the decoder and pipeline faked."""
    clips = tmp_path / "clips"
    write_miv(clips)
    cache = tmp_path / "cache"
    conversions = ConversionStore(
        cache, TmivInstall(tmp_path / "tmiv"), device="cpu",
        run=partial(run_conversion, spatializer=fake_spatializer,
                    probe=lambda path: make_video_info(),
                    # slow enough per frame that a cancel lands between two of them
                    run=fake_pipeline(source_frames=2, delay=FAKE_FRAME_S)),
    )
    library = Library([clips], cache / "uploads", conversions.converted_dir)
    previews = PreviewStore(
        cache, TmivInstall(tmp_path / "tmiv"),
        build=partial(build_preview, decode=fake_decoder(frames),
                      probe=lambda *args, **kwargs: 2),
    )
    store = JobStore()
    config = MediaServerConfig(store, None, "cpu", "fp16", None, library=library,
                               previews=previews, conversions=conversions)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(SpatialMediaHandler, config=config))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield Client(f"http://127.0.0.1:{httpd.server_address[1]}"), library
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        store.clear()


def test_the_video_page_and_its_assets_are_served(served):
    """/video is the video page; its JS and CSS come from /video/static."""
    client, _library = served
    status, headers, body = client.request("/video")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert b"Spatial Video Preview" in body

    for name, prefix in (("video.js", ""), ("video.css", "text/css"), ("modes.css", "text/css")):
        status, headers, body = client.request(f"/video/static/{name}")
        assert status == 200 and body, name
        assert headers["Content-Type"].startswith(prefix), name

    for path in ("/video/static/..%2fserver.py", "/video/static/../server.py"):
        assert client.request(path)[0] == 404, path


def test_the_photo_page_gains_the_mode_switch_and_keeps_working(served):
    """`/` is still Stage 1's page and API, now with a link to the video page."""
    client, _library = served
    status, _headers, body = client.request("/")
    assert status == 200
    assert b'id="dropzone"' in body and b"Spatial Photo Preview" in body
    assert b'href="/video"' in body and b"/video/static/modes.css" in body

    assert client.request("/static/app.js")[0] == 200
    status, config = client.json("/api/config")
    assert status == 200 and "defaults" in config


def test_the_library_lists_clips_with_their_preview_state(served):
    """Each .miv comes with its manifest details, and no preview until one is asked for."""
    client, _library = served
    status, library = client.json("/api/miv/library")

    assert status == 200
    assert library["tmiv"]["ok"] is True
    [source] = library["sources"]
    assert source["name"] == "clip.miv" and source["frame_count"] == 3
    assert source["preview"] is None


def test_a_preview_builds_and_serves_both_videos(served):
    """POST starts the build, polling reaches done, and both mosaics download."""
    client, library = served
    [source] = library.sources()

    status, payload = client.json(f"/api/miv/sources/{source.source_id}/preview", "POST", b"")
    assert status == 202 and payload["source"]["id"] == source.source_id
    preview = client.wait_for_preview(source.source_id)
    assert preview["state"] == "done", preview["error"]
    assert preview["meta"]["frame_count"] == 3

    for layer in ("color", "depth"):
        status, headers, body = client.request(f"/api/miv/previews/{source.source_id}/{layer}.mp4")
        assert status == 200 and headers["Content-Type"] == "video/mp4", layer
        assert body[4:8] == b"ftyp", layer

    _status, headers, _body = client.request(
        f"/api/miv/previews/{source.source_id}/color.mp4?download=1")
    assert 'filename="clip_color.mp4"' in headers["Content-Disposition"]

    _status, listed = client.json("/api/miv/library")
    assert listed["sources"][0]["preview"]["state"] == "done"


def test_an_uploaded_miv_joins_the_library_and_is_previewed(served):
    """Uploading a bare .miv (no manifest) adds it and builds its preview."""
    client, _library = served
    status, payload = client.json("/api/miv/upload?filename=phone.miv", "POST", b"\x00MIV" * 32)

    assert status == 202, payload
    preview = client.wait_for_preview(payload["source"]["id"])
    assert preview["state"] == "done", preview["error"]
    assert preview["meta"]["frame_count"] == 2  # probed, as there is no manifest

    _status, listed = client.json("/api/miv/library")
    uploaded = [s for s in listed["sources"] if s["name"] == "phone.miv"]
    assert uploaded and uploaded[0]["uploaded"] is True


def test_an_uploaded_video_is_converted_and_then_previewed(served):
    """The whole video path end to end: upload, convert, and the .miv it wrote previews."""
    client, library = served
    status, payload = client.json("/api/video/upload?filename=holiday.mp4&max_frames=2",
                                  "POST", b"\x00\x00\x00 ftypisom" * 40)
    assert status == 202
    conversion = payload["conversion"]
    assert conversion["name"] == "holiday.mp4" and conversion["options"]["max_frames"] == 2

    finished = client.wait_for_conversion(conversion["id"])
    assert finished["state"] == "done", finished["error"]
    assert finished["miv"]["frame_count"] == 2

    _status, listed = client.json("/api/miv/library")
    converted = [source for source in listed["sources"] if source["id"] == finished["source_id"]]
    assert converted and converted[0]["origin"] == "converted"
    assert converted[0]["name"] == "holiday.miv"
    assert [c["id"] for c in listed["conversions"]] == [conversion["id"]]

    client.json(f"/api/miv/sources/{finished['source_id']}/preview", "POST", b"")
    assert client.wait_for_preview(finished["source_id"])["state"] == "done"


def test_the_library_says_what_the_converter_accepts(served):
    """The browser learns the suffixes, the size cap and the defaults from the server."""
    client, _library = served
    _status, listed = client.json("/api/miv/library")
    convert = listed["convert"]
    assert ".mp4" in convert["suffixes"] and ".miv" not in convert["suffixes"]
    assert convert["defaults"]["max_frames"] is None  # no cap: the whole video
    assert None in convert["view_heights"] and 720 in convert["view_heights"]
    assert convert["min_angle_deg"] < convert["defaults"]["angle_deg"] <= convert["max_angle_deg"]
    assert convert["max_upload_mb"] > 0 and convert["device"] == "cpu"


def test_a_conversion_can_be_stopped_from_the_browser(served):
    """Cancel reaches the run, which stops instead of writing a clip."""
    client, _library = served
    _status, payload = client.json("/api/video/upload?filename=long.mp4", "POST", b"mp4" * 40)
    conversion_id = payload["conversion"]["id"]
    status, cancelled = client.json(f"/api/video/conversions/{conversion_id}/cancel", "POST", b"")
    assert status == 200 and cancelled["cancelling"] is True

    finished = client.wait_for_conversion(conversion_id)
    assert finished["state"] == "cancelled"
    _status, listed = client.json("/api/miv/library")
    assert [s for s in listed["sources"] if s["origin"] == "converted"] == []


def test_a_bad_conversion_option_is_refused_before_the_upload(served):
    """The options are parsed first, so a typo does not cost a whole upload."""
    client, _library = served
    status, payload = client.json("/api/video/upload?filename=clip.mp4&max_frames=nope",
                                  "POST", b"mp4" * 40)
    assert status == 400 and "whole number" in payload["error"]


@pytest.mark.parametrize(
    ("path", "method", "body", "expected"),
    [
        ("/api/miv/previews/feedfacecafe", "GET", None, 404),
        ("/api/miv/sources/feedfacecafe/preview", "POST", b"", 404),
        ("/api/miv/upload?filename=clip.h5", "POST", b"data", 400),
        ("/api/miv/upload?filename=clip.miv", "POST", b"", 400),
        ("/api/miv/nope", "GET", None, 404),
        ("/api/video/upload?filename=clip.miv", "POST", b"data", 400),
        ("/api/video/upload?filename=clip.mp4", "POST", b"", 400),
        ("/api/video/conversions/feedfacecafe", "GET", None, 404),
        ("/api/video/conversions/feedfacecafe/cancel", "POST", b"", 404),
        ("/api/video/nope", "GET", None, 404),
    ],
)
def test_bad_requests_get_json_errors(served, path, method, body, expected):
    """Unknown ids, wrong files and unknown endpoints are clean JSON errors."""
    client, _library = served
    status, payload = client.json(path, method, body)
    assert status == expected and "error" in payload


def test_preview_files_are_404_before_anything_is_built(served):
    """Asking for a video of a clip that was never previewed does not start work."""
    client, library = served
    [source] = library.sources()
    status, _payload = client.json(f"/api/miv/previews/{source.source_id}/color.mp4")
    assert status == 404


# -------------------------------------------------------------- real TMIV


@pytest.mark.slow
def test_a_real_miv_file_previews_with_and_without_its_manifest(tmp_path):
    """Encode with the patched TMIV, then preview with the real TmivDecoder and TmivParser."""
    from sharp_video.miv.encoder import MIVEncoderConfig, encode_sequence
    from sharp_video.miv.tmiv import TmivNotFoundError, find_tmiv
    from sharp_video.sequence_io import SequenceReader, SequenceWriter
    from synthetic import make_info

    try:
        tmiv = find_tmiv()
    except TmivNotFoundError as exc:
        pytest.skip(str(exc))

    sequence = make_frames(4, width=64, height=64, drift=0.3, motion=0.2, tilt=0.4)
    path = tmp_path / "seq.h5"
    with SequenceWriter(path, make_info(64, 64)) as writer:
        for frame in sequence:
            writer.write_frame(frame)
    config = MIVEncoderConfig(work_dir=tmp_path / "enc", tmiv=tmiv, intra_period=16, threads=2)
    miv = encode_sequence(SequenceReader(path), config, tmp_path / "clip.miv").path

    with_manifest = Preview(MivSource.from_path(miv), tmp_path / "a")
    build_preview(with_manifest, tmiv=tmiv)
    assert with_manifest.state == "done", with_manifest.error

    Path(f"{miv}.json").unlink()
    bare = Preview(MivSource.from_path(miv), tmp_path / "b")
    build_preview(bare, tmiv=tmiv)
    assert bare.state == "done", bare.error

    for preview in (with_manifest, bare):
        assert preview.meta["frame_count"] == 4
        assert preview.meta["camera_update_frames"] == [0, 1, 2, 3]
        color = read_video(preview.file("color.mp4"))
        assert len(color) == 4 and color[0].shape == (192, 192, 3)
