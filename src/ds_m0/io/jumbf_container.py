"""JUMBF (ISO/IEC 19566-5) box codec and its embedding in JPEG APP11 segments.

Layout written by this module (everything else sees only `bytes`):

    JPEG:  SOI, APP0(JFIF)..., APP11 packets carrying ONE 'jumb' superbox, ...image data...
    jumb  "ds-image"  (jumd UUID = DS content type)
      jumb "header"        jumd(bidb UUID)  + bidb
      jumb "metadata"      jumd(json UUID)  + json
      jumb "layer0_depth"  jumd(bidb UUID)  + bidb      ... one jumb per payload
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..errors import FormatError

UUID_DS = uuid.UUID("44532d49-4d41-4745-0000-000000000001").bytes  # "DS-IMAGE" + 1
UUID_JSON = uuid.UUID("6a736f6e-0011-0010-8000-00aa00389b71").bytes  # ISO 19566-5 JSON content type
UUID_BIDB = uuid.UUID("40cb0c32-bb8a-489d-a70b-2ad6f47f4369").bytes  # ISO 19566-5 embedded file/binary
ROOT_LABEL = "ds-image"
MAGIC = b"DSM0"

_APP11 = b"\xff\xeb"
_MAX_CHUNK = 65535 - 2 - 2 - 2 - 4 - 8  # Le(2) CI(2) En(2) Z(4) LBox/TBox(8)
_TYPE_FOR = {"json": b"json", "bidb": b"bidb"}


def make_box(box_type: bytes, payload: bytes) -> bytes:
    if len(payload) + 8 > 0xFFFFFFFF:
        raise FormatError("box too large for a 32-bit LBox")
    return struct.pack(">I", len(payload) + 8) + box_type + payload


def parse_boxes(data: bytes) -> list[tuple[bytes, bytes]]:
    """Split `data` into (type, payload) boxes. Strict: truncation or junk is an error."""
    boxes, pos = [], 0
    while pos < len(data):
        if pos + 8 > len(data):
            raise FormatError("truncated JUMBF box header")
        (lbox,) = struct.unpack(">I", data[pos:pos + 4])
        tbox = data[pos + 4:pos + 8]
        header = 8
        if lbox == 1:
            if pos + 16 > len(data):
                raise FormatError("truncated JUMBF XLBox")
            (lbox,) = struct.unpack(">Q", data[pos + 8:pos + 16])
            header = 16
        elif lbox == 0:
            lbox = len(data) - pos
        if lbox < header or pos + lbox > len(data):
            raise FormatError(f"invalid JUMBF box length {lbox} for box {tbox!r}")
        boxes.append((tbox, data[pos + header:pos + lbox]))
        pos += lbox
    return boxes


def _jumd(content_uuid: bytes, label: str) -> bytes:
    return make_box(b"jumd", content_uuid + b"\x03" + label.encode("utf-8") + b"\x00")


def _parse_jumb(payload: bytes) -> tuple[bytes, str, list[tuple[bytes, bytes]]]:
    boxes = parse_boxes(payload)
    if not boxes or boxes[0][0] != b"jumd":
        raise FormatError("jumb superbox does not start with a jumd description box")
    desc = boxes[0][1]
    if len(desc) < 18 or desc[-1:] != b"\x00":
        raise FormatError("malformed jumd description box")
    try:
        label = desc[17:-1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FormatError("jumd label is not UTF-8") from exc
    return desc[:16], label, boxes[1:]


@dataclass
class ContainerContents:
    primary_jpeg: bytes  # the JPEG with every DS APP11 segment removed
    payloads: dict[str, tuple[str, bytes]]  # label -> (content type 'json'|'bidb', bytes)
    app11_bytes: int  # total size of the DS APP11 segments in the file


class DSContainerWriter:
    def __init__(self) -> None:
        self._jpeg: bytes | None = None
        self._payloads: list[tuple[str, str, bytes]] = []

    def create(self) -> "DSContainerWriter":
        self._jpeg, self._payloads = None, []
        return self

    def add_primary_jpeg(self, jpeg: bytes) -> None:
        if jpeg[:2] != b"\xff\xd8":
            raise FormatError("primary image is not a JPEG (no SOI marker)")
        self._jpeg = jpeg

    def add_ds_payload(self, label: str, content_type: str, data: bytes) -> None:
        if content_type not in _TYPE_FOR:
            raise FormatError(f"unknown JUMBF content type {content_type!r}")
        if any(label == p[0] for p in self._payloads):
            raise FormatError(f"duplicate payload label {label!r}")
        self._payloads.append((label, content_type, data))

    def finalize(self) -> bytes:
        if self._jpeg is None:
            raise FormatError("no primary JPEG added")
        children = [
            make_box(b"jumb", _jumd(UUID_BIDB, "header") + make_box(b"bidb", MAGIC))
        ]
        for label, ctype, data in self._payloads:
            uid = UUID_JSON if ctype == "json" else UUID_BIDB
            children.append(make_box(b"jumb", _jumd(uid, label) + make_box(_TYPE_FOR[ctype], data)))
        root = make_box(b"jumb", _jumd(UUID_DS, ROOT_LABEL) + b"".join(children))
        segments = _segment_box(root, instance=1)
        at = _insertion_offset(self._jpeg)
        return self._jpeg[:at] + b"".join(segments) + self._jpeg[at:]


def _segment_box(box: bytes, instance: int) -> list[bytes]:
    hdr, body = box[:8], box[8:]
    segs, pos, z = [], 0, 1
    while True:
        chunk = body[pos:pos + _MAX_CHUNK]
        pos += len(chunk)
        payload = b"JP" + struct.pack(">HI", instance, z) + hdr + chunk
        segs.append(_APP11 + struct.pack(">H", len(payload) + 2) + payload)
        z += 1
        if pos >= len(body):
            return segs


def _segments(data: bytes) -> list[tuple[int, int, int]]:
    """(marker, start, end) for every length-bearing segment before SOS."""
    if data[:2] != b"\xff\xd8":
        raise FormatError("not a JPEG: missing SOI marker")
    out, pos = [], 2
    while pos + 2 <= len(data):
        if data[pos] != 0xFF:
            raise FormatError(f"corrupt JPEG: expected marker at offset {pos}")
        marker = data[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            pos += 2
            continue
        if marker in (0xDA, 0xD9):
            return out
        if pos + 4 > len(data):
            raise FormatError("truncated JPEG segment header")
        (length,) = struct.unpack(">H", data[pos + 2:pos + 4])
        end = pos + 2 + length
        if length < 2 or end > len(data):
            raise FormatError("truncated JPEG segment")
        out.append((marker, pos, end))
        pos = end
    raise FormatError("truncated JPEG: no start-of-scan marker")


def _insertion_offset(jpeg: bytes) -> int:
    at = 2
    for marker, start, end in _segments(jpeg):
        if start != at or marker not in (0xE0, 0xE1):
            break
        at = end
    return at


class DSContainerReader:
    def __init__(self) -> None:
        self._contents: ContainerContents | None = None

    def open(self, path: str | Path) -> "DSContainerReader":
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise FormatError(f"cannot read {path}: {exc}") from exc
        self._contents = self.parse(data)
        return self

    def read_primary_jpeg(self) -> bytes:
        return self._need().primary_jpeg

    def read_ds_payload(self) -> dict[str, tuple[str, bytes]]:
        return self._need().payloads

    def contents(self) -> ContainerContents:
        return self._need()

    def close(self) -> None:
        self._contents = None

    def _need(self) -> ContainerContents:
        if self._contents is None:
            raise FormatError("container is not open")
        return self._contents

    @staticmethod
    def parse(data: bytes) -> ContainerContents:
        segs = [s for s in _segments(data) if s[0] == 0xEB and data[s[1] + 4:s[1] + 6] == b"JP"]
        if not segs:
            raise FormatError("no DS payload: file has no JUMBF (APP11) segments")
        packets: dict[int, list[tuple[int, bytes, bytes]]] = {}
        for _, start, end in segs:
            if end - start < 20:
                raise FormatError("truncated JUMBF packet")
            en, z = struct.unpack(">HI", data[start + 6:start + 12])
            packets.setdefault(en, []).append((z, data[start + 12:start + 20], data[start + 20:end]))
        if 1 not in packets:
            raise FormatError("DS JUMBF box instance 1 is missing")
        root = _reassemble(packets[1])
        stripped = data
        for _, start, end in sorted(segs, reverse=True):
            stripped = stripped[:start] + stripped[end:]
        app11 = sum(end - start for _, start, end in segs)

        (tbox, payload), = parse_boxes(root)
        if tbox != b"jumb":
            raise FormatError("DS payload root is not a jumb superbox")
        uid, label, children = _parse_jumb(payload)
        if uid != UUID_DS or label != ROOT_LABEL:
            raise FormatError("JUMBF payload is not a DS-Image payload")
        payloads: dict[str, tuple[str, bytes]] = {}
        seen_header = False
        for ctype4, cpayload in children:
            if ctype4 != b"jumb":
                raise FormatError(f"unexpected box {ctype4!r} in DS payload")
            cuid, clabel, content = _parse_jumb(cpayload)
            if len(content) != 1:
                raise FormatError(f"payload {clabel!r} must contain exactly one content box")
            kind, body = content[0]
            if clabel == "header":
                if kind != b"bidb" or body[:4] != MAGIC:
                    raise FormatError("bad DS header (magic mismatch)")
                seen_header = True
                continue
            if kind not in (b"json", b"bidb"):
                raise FormatError(f"payload {clabel!r} has unknown content box {kind!r}")
            if clabel in payloads:
                raise FormatError(f"duplicate payload {clabel!r}")
            payloads[clabel] = (kind.decode(), body)
        if not seen_header:
            raise FormatError("DS header box missing")
        return ContainerContents(stripped, payloads, app11)


def _reassemble(packets: list[tuple[int, bytes, bytes]]) -> bytes:
    packets = sorted(packets, key=lambda p: p[0])
    if [p[0] for p in packets] != list(range(1, len(packets) + 1)):
        raise FormatError("JUMBF packet sequence is incomplete or duplicated (truncated file?)")
    header = packets[0][1]
    if any(p[1] != header for p in packets):
        raise FormatError("JUMBF packet box headers disagree")
    box = header + b"".join(p[2] for p in packets)
    (declared,) = struct.unpack(">I", header[:4])
    if declared != len(box):
        raise FormatError(f"JUMBF box declares {declared} bytes but {len(box)} were found (truncated file?)")
    return box
