"""Decoding an MIV file with the independent reference decoder (TmivDecoder).

Used only for validation (spec §38: "an independent MIV decoder can
successfully parse/decode the resulting file"); Stage 2 does not implement an
MIV decoder. TmivDecoder is asked to reconstruct every source view (texture,
geometry, occupancy) and to write the sequence configuration -- i.e. the
decoded camera parameters -- for every frame at which it changes.
`TmivParser` additionally dumps the bitstream's high-level syntax.

Decoded geometry is 10-bit normalized disparity relative to each view's
decoded `Depth_range` (TMIV adapts the range to the content), and validity
comes from the decoded occupancy map.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import camera_from_tmiv_json
from .depth import geometry_to_depth
from .tmiv import TmivInstall, find_tmiv, run_logged
from .tmiv_config import CONTENT_ID, RATE_ID, decoder_config
from .yuv import frame_nbytes, read_yuv420_frame, yuv420_to_rgb

_FORMAT_BIT_DEPTH = {"yuv420p": 8, "yuv420p10le": 10, "yuv420p12le": 12, "yuv420p16le": 16,
                     "gray": 8, "gray10le": 10, "gray12le": 12, "gray16le": 16}
_FILE_RE = re.compile(r"^(\d+)_(tex|geo|occ)_(\d+)x(\d+)_(\w+)\.yuv$")


@dataclass
class DecodedView:
    """One reconstructed view in Stage 1 conventions."""

    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    R: np.ndarray
    C: np.ndarray


@dataclass
class _ViewFile:
    path: Path
    width: int
    height: int
    bit_depth: int
    chroma: bool  # 4:2:0 file (True) or luma-only "gray" file (False)

    def read_luma(self, handle) -> np.ndarray:
        """Next frame's luma plane (skipping chroma for 4:2:0 files)."""
        if self.chroma:
            return read_yuv420_frame(handle, self.width, self.height, self.bit_depth)[0]
        dtype = np.uint8 if self.bit_depth <= 8 else np.dtype("<u2")
        nbytes = self.width * self.height * np.dtype(dtype).itemsize
        data = handle.read(nbytes)
        if len(data) < nbytes:
            raise EOFError(f"end of {self.path.name}")
        return np.frombuffer(data, dtype=dtype).astype(np.uint16).reshape(
            self.height, self.width)


class DecodedMiv:
    """TmivDecoder's reconstruction of an MIV file, read back frame by frame."""

    def __init__(self, decoded_dir: Path, frame_count: int, parser_dump: Path | None = None):
        """Read TmivDecoder's output in `decoded_dir`: per-frame cameras and view files."""
        self.decoded_dir = Path(decoded_dir)
        self.frame_count = frame_count
        self.parser_dump = parser_dump
        self._configs = {
            int(path.stem.split("_")[1]): json.loads(path.read_text())
            for path in sorted(self.decoded_dir.glob("seq_*.json"))
        }
        if 0 not in self._configs:
            raise RuntimeError(f"TmivDecoder wrote no sequence configuration in {decoded_dir}")
        first = self._configs[0]
        self.fps = float(first["Fps"])
        # MIV identifies views by index/ID, not by name: TmivDecoder calls them pv00, pv01, ...
        # in view-index order, which is the v0..v8 order the encoder coded them in.
        self.decoded_view_names = [camera["Name"] for camera in first["cameras"]]
        self.view_names = [f"v{i}" for i in range(len(self.decoded_view_names))]
        width, height = first["cameras"][0]["Resolution"]
        self.resolution = (int(width), int(height))
        self._files: dict[tuple[int, str], _ViewFile] = {}
        for path in self.decoded_dir.glob("*.yuv"):
            match = _FILE_RE.match(path.name)
            if match:
                index, kind, w, h, fmt = match.groups()
                self._files[(int(index), kind)] = _ViewFile(
                    path, int(w), int(h), _FORMAT_BIT_DEPTH[fmt], not fmt.startswith("gray"))

    @property
    def camera_update_frames(self) -> list[int]:
        """Frames at which the decoded camera parameters changed (0 = initial)."""
        return sorted(self._configs)

    def config_at(self, frame_index: int) -> dict:
        """The decoded sequence configuration in effect at `frame_index`."""
        return self._configs[max(k for k in self._configs if k <= frame_index)]

    def frames(self) -> Iterator[dict[str, DecodedView]]:
        """Yield `{view_name: DecodedView}` for every decoded frame, in order."""
        handles = {key: open(f.path, "rb") for key, f in self._files.items()}
        try:
            for t in range(self.frame_count):
                cameras = {c["Name"]: c for c in self.config_at(t)["cameras"]}
                yield {
                    name: self._read_view(handles, index, cameras[decoded_name])
                    for index, (name, decoded_name) in enumerate(
                        zip(self.view_names, self.decoded_view_names))
                }
        finally:
            for handle in handles.values():
                handle.close()

    def _read_view(self, handles, index: int, camera: dict) -> DecodedView:
        tex = self._files[(index, "tex")]
        rgb = yuv420_to_rgb(*read_yuv420_frame(handles[(index, "tex")], tex.width, tex.height,
                                               tex.bit_depth), bit_depth=tex.bit_depth)
        geo = self._files[(index, "geo")]
        samples = geo.read_luma(handles[(index, "geo")])
        near, far = camera["Depth_range"]
        depth, mask = geometry_to_depth(samples, near, far, geo.bit_depth)
        if (index, "occ") in self._files:
            occ = self._files[(index, "occ")]
            mask = (occ.read_luma(handles[(index, "occ")]) != 0).astype(np.uint8)
            depth = np.where(mask == 1, depth, 0.0).astype(np.float32)
        K, R, C = camera_from_tmiv_json(camera)
        return DecodedView(rgb=rgb, depth=depth, mask=mask, K=K, R=R, C=C)


def decode_miv(
    miv_path: str | Path,
    frame_count: int,
    work_dir: str | Path,
    tmiv: TmivInstall | None = None,
    runner=run_logged,
) -> DecodedMiv:
    """Decode `miv_path` with TmivDecoder into `work_dir/decoded` and return the result."""
    tmiv = tmiv or find_tmiv()
    work_dir = Path(work_dir).resolve()
    (work_dir / "decoded").mkdir(parents=True, exist_ok=True)
    miv_path = Path(miv_path).resolve()

    config_path = work_dir / "decoder.json"
    config_path.write_text(json.dumps(decoder_config(miv_path), indent=2))
    runner([
        tmiv.exe("TmivDecoder"), "-c", config_path,
        "-p", "inputDirectory", work_dir, "-p", "outputDirectory", work_dir,
        "-s", CONTENT_ID, "-r", RATE_ID, "-n", str(frame_count), "-N", str(frame_count),
    ], work_dir / "logs" / "decode.log", cwd=work_dir)

    parser_dump = work_dir / "parsed.hls"
    runner([tmiv.exe("TmivParser"), "-b", miv_path, "-o", parser_dump],
           work_dir / "logs" / "parse.log", cwd=work_dir)
    return DecodedMiv(work_dir / "decoded", frame_count, parser_dump)


def frame_count_from_parser_dump(text: str) -> int:
    """Frames in a bitstream, from TmivParser's dump: one atlas tile header per atlas per frame."""
    atlases = re.search(r"^vps_atlas_count_minus1=(\d+)", text, re.MULTILINE)
    headers = len(re.findall(r"^ath_id=", text, re.MULTILINE))
    if atlases is None or headers == 0:
        raise ValueError("TmivParser found no V3C parameter set or atlas frames")
    return headers // (int(atlases[1]) + 1)


def probe_frame_count(
    miv_path: str | Path,
    work_dir: str | Path,
    tmiv: TmivInstall | None = None,
    runner=run_logged,
) -> int:
    """Count the frames of `miv_path` with TmivParser, which parses without decoding.

    TmivDecoder needs the frame count up front; the Stage 2 manifest records it,
    so this is for a .miv that arrives without its manifest.
    """
    tmiv = tmiv or find_tmiv()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    dump = work_dir / "probe.hls"
    runner([tmiv.exe("TmivParser"), "-b", Path(miv_path).resolve(), "-o", dump],
           work_dir / "logs" / "probe.log", cwd=work_dir)
    return frame_count_from_parser_dump(dump.read_text())


def decoded_frames_written(decoded_dir: str | Path) -> int:
    """How many frames TmivDecoder has written to `decoded_dir` so far (for progress).

    TmivDecoder writes frame by frame, so view 0's texture file grows one frame at a time.
    """
    for path in Path(decoded_dir).glob("00_tex_*.yuv"):
        match = _FILE_RE.match(path.name)
        if match:
            _, _, width, height, fmt = match.groups()
            return path.stat().st_size // frame_nbytes(int(width), int(height),
                                                       _FORMAT_BIT_DEPTH[fmt])
    return 0
