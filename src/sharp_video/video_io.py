"""Video input: probing and presentation-order decoding to RGB (spec §8, §9).

Uses PyAV, which exposes presentation timestamps, time bases, pixel formats
and colour metadata directly. Initial target is H.264/H.265 (in MP4 or any
container FFmpeg reads); to support another codec, add its FFmpeg decoder name
to `SUPPORTED_CODECS`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from av.video.reformatter import Colorspace

from .scheduler import FrameSelection

SUPPORTED_CODECS = frozenset({"h264", "hevc"})

# FFmpeg AVColorSpace values -> swscale YUV->RGB matrices.
_AVCOL_SPC_TO_SWS = {
    1: Colorspace.ITU709,  # BT.709
    4: Colorspace.FCC,
    5: Colorspace.ITU601,  # BT.470BG
    6: Colorspace.ITU601,  # SMPTE 170M
    7: Colorspace.SMPTE240M,
    9: Colorspace.ITU624,  # BT.2020 NCL
}
_AVCOL_SPC_UNSPECIFIED = 2
_HD_HEIGHT = 720  # untagged content at or above this height is assumed BT.709


class UnsupportedVideoError(ValueError):
    """The input container/codec is not one this tool decodes."""


@dataclass
class VideoInfo:
    """What the input parser reports about a video (spec §8)."""

    path: str
    container: str
    codec: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # seconds
    time_base: str  # e.g. "1/30000"
    pix_fmt: str
    color_primaries: str
    color_trc: str
    colorspace: str
    color_range: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class DecodedFrame:
    """One decoded frame, in presentation order."""

    index: int  # presentation-order index in the source
    pts: int  # presentation timestamp in the stream's time base
    timestamp: float  # seconds
    rgb: np.ndarray  # [H, W, 3] uint8


def _open_video_stream(container: av.container.InputContainer, path: Path):
    if not container.streams.video:
        raise UnsupportedVideoError(f"{path} has no video stream")
    stream = container.streams.video[0]
    codec = stream.codec_context.name
    if codec not in SUPPORTED_CODECS:
        raise UnsupportedVideoError(
            f"{path}: codec {codec!r} is not supported (supported: {sorted(SUPPORTED_CODECS)})"
        )
    return stream


def _enum_name(value) -> str:
    if value is None:
        return "unknown"
    return getattr(value, "name", str(value))


def probe_video(path: str | Path) -> VideoInfo:
    """Read the input's geometry, timing and colour information."""
    path = Path(path)
    with av.open(str(path)) as container:
        stream = _open_video_stream(container, path)
        ctx = stream.codec_context
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        fps = float(rate) if rate else 0.0

        frame_count = stream.frames
        if not frame_count:
            frame_count = sum(1 for packet in container.demux(stream) if packet.size)
            container.seek(0)

        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        else:
            duration = frame_count / fps if fps else 0.0

        first = next(container.decode(stream), None)

    return VideoInfo(
        path=str(path),
        container=container.format.name,
        codec=ctx.name,
        width=ctx.width,
        height=ctx.height,
        fps=fps,
        frame_count=int(frame_count),
        duration=duration,
        time_base=str(stream.time_base),
        pix_fmt=ctx.pix_fmt or "",
        color_primaries=_enum_name(getattr(first, "color_primaries", None)),
        color_trc=_enum_name(getattr(first, "color_trc", None)),
        colorspace=_enum_name(getattr(first, "colorspace", None)),
        color_range=_enum_name(getattr(first, "color_range", None)),
    )


def _to_rgb(frame: av.VideoFrame) -> np.ndarray:
    spc = int(frame.colorspace) if frame.colorspace is not None else _AVCOL_SPC_UNSPECIFIED
    matrix = _AVCOL_SPC_TO_SWS.get(spc)
    if matrix is None:
        matrix = Colorspace.ITU709 if frame.height >= _HD_HEIGHT else Colorspace.ITU601
    return frame.reformat(format="rgb24", src_colorspace=matrix).to_ndarray()


def iter_frames(
    path: str | Path, selection: FrameSelection | None = None
) -> Iterator[DecodedFrame]:
    """Decode the selected frames of `path` in presentation order.

    Frame indices count decoded frames in presentation order from 0; the
    original PTS and timestamp are kept (a selection starting at frame 30 has
    timestamps starting at ~1 s, not 0).
    """
    path = Path(path)
    selection = (selection or FrameSelection()).validate()
    with av.open(str(path)) as container:
        stream = _open_video_stream(container, path)
        stream.thread_type = "AUTO"
        selected = 0
        last_pts = None
        for index, frame in enumerate(container.decode(stream)):
            if selection.exhausted(index, selected):
                break
            if frame.pts is not None and last_pts is not None and frame.pts <= last_pts:
                raise ValueError(
                    f"{path}: decoder produced non-increasing PTS {frame.pts} after {last_pts}"
                )
            last_pts = frame.pts
            if not selection.contains(index, selected):
                continue
            selected += 1
            yield DecodedFrame(
                index=index,
                pts=int(frame.pts),
                timestamp=float(frame.pts * stream.time_base),
                rgb=_to_rgb(frame),
            )
