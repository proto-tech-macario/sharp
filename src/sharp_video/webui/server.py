"""The spatial-media web UI: Stage 1's photo page plus a video page for MIV files.

Stage 1's server is extended, not edited (Stage 2 leaves `src/sharp_spatialize`
untouched): its handler is subclassed, and its page gets a Photo | Video switch
as it is served. So `/` is still the spatial-photo UI and runs Stage 1 jobs as
before, and `/video` converts ordinary videos to MIV and previews MIV files:

    GET  /video                                the video page
    GET  /api/miv/library                      .miv files on offer, previews, conversions
    POST /api/miv/sources/<id>/preview         build the preview (idempotent)
    POST /api/miv/upload?filename=<name>.miv   add a .miv from the browser, then build it
    GET  /api/miv/previews/<id>                build progress; metadata once done
    GET  /api/miv/previews/<id>/<layer>.mp4    a 3x3 mosaic video, layer = color | depth

    POST /api/video/upload?filename=<name>.mp4&<options>   convert a video to MIV
    GET  /api/video/conversions/<id>                       conversion progress
    POST /api/video/conversions/<id>/cancel                stop it after the current frame

A finished conversion reports the `source_id` of the .miv it wrote, which the
browser then previews through the /api/miv endpoints like any other clip.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import shutil
import threading
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from sharp_spatialize.webui import server as photo_server
from sharp_spatialize.webui.jobs import JobStore
from sharp_spatialize.webui.server import ServerConfig, SpatialPhotoHandler

from ..miv.tmiv import TmivInstall, TmivNotFoundError, find_tmiv
from .conversions import (
    MAX_ANGLE_DEG,
    SLIDER_MIN_ANGLE_DEG,
    VIDEO_SUFFIXES,
    VIEW_HEIGHTS,
    ConversionOptions,
    ConversionStore,
)
from .previews import CHUNK_BYTES, PREVIEW_FILES, Library, MivSource, PreviewStore

STATIC_DIR = Path(__file__).parent / "static"
#: This source checkout; its `.tmiv` is the last place TMIV is looked for.
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_HOST = photo_server.DEFAULT_HOST
DEFAULT_PORT = 8738
MAX_MIV_UPLOAD_BYTES = 4 * 1024**3
MAX_VIDEO_UPLOAD_BYTES = 8 * 1024**3

_MODE_SWITCH = (
    '<nav class="modes" aria-label="Mode">'
    '<a href="/" class="active" aria-current="page">Photo</a><a href="/video">Video</a></nav>'
)
_MODES_STYLESHEET = '<link rel="stylesheet" href="/video/static/modes.css">'


def _segments(path: str) -> list[str]:
    return [segment for segment in path.split("/") if segment]


class SpatialMediaHandler(SpatialPhotoHandler):
    """Stage 1's handler plus the video page and the /api/miv and /api/video endpoints."""

    server_version = "sharp-video-webui"

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        """Route the video page, /api/miv and /api/video; everything else is Stage 1's."""
        parsed = urlparse(self.path)
        segments = _segments(parsed.path)
        try:
            if not segments:
                return self._serve_photo_page()
            if segments == ["video"]:
                return self._serve_asset("video.html")
            if segments[:2] == ["video", "static"] and len(segments) == 3:
                return self._serve_asset(segments[2])
            if segments[:2] == ["api", "miv"]:
                return self._route_miv_get(segments[2:], parse_qs(parsed.query))
            if segments[:2] == ["api", "video"]:
                return self._route_video_get(segments[2:])
        except ConnectionError:  # pragma: no cover - client navigated away mid-response
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        """Route /api/miv and /api/video POSTs; everything else is Stage 1's."""
        parsed = urlparse(self.path)
        segments = _segments(parsed.path)
        if segments[:2] == ["api", "miv"]:
            return self._route_miv_post(segments[2:], parse_qs(parsed.query))
        if segments[:2] == ["api", "video"]:
            return self._route_video_post(segments[2:], parse_qs(parsed.query))
        super().do_POST()

    # -- pages ------------------------------------------------------------

    def _serve_photo_page(self) -> None:
        """Stage 1's page with the Photo | Video switch added (the file itself is untouched)."""
        html = (photo_server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
        chips = '<div class="chips">'
        if chips in html and "</head>" in html:
            html = html.replace(chips, f"{_MODE_SWITCH}\n    {chips}", 1)
            html = html.replace("</head>", f"{_MODES_STYLESHEET}\n</head>", 1)
        self._send(HTTPStatus.OK, html.encode(), "text/html; charset=utf-8")

    def _serve_asset(self, name: str) -> None:
        path = (STATIC_DIR / name).resolve()
        if path.parent != STATIC_DIR.resolve() or not path.is_file():
            return self._send_error_json(HTTPStatus.NOT_FOUND, f"no asset {name}")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type.endswith("javascript"):
            content_type += "; charset=utf-8"
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    def _send_file(self, path: Path, content_type: str, download_name: str | None = None) -> None:
        """Stream a (possibly large) file instead of buffering it."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            safe = download_name.replace('"', "")
            self.send_header("Content-Disposition", f'attachment; filename="{safe}"')
        self.end_headers()
        if self.command != "HEAD":
            with open(path, "rb") as handle:
                shutil.copyfileobj(handle, self.wfile, CHUNK_BYTES)

    # -- /api/miv ---------------------------------------------------------

    def _route_miv_get(self, segments: list[str], query: dict[str, list[str]]) -> None:
        if segments == ["library"]:
            return self._send_json(HTTPStatus.OK, self.config.describe_library())

        if len(segments) in (2, 3) and segments[0] == "previews":
            source = self.config.library.get(segments[1])
            if source is None:
                return self._send_error_json(HTTPStatus.NOT_FOUND,
                                             "unknown .miv file (moved, changed or deleted?)")
            preview = self.config.previews.get(source)
            if preview is None:
                return self._send_error_json(HTTPStatus.NOT_FOUND, "no preview built yet")
            if len(segments) == 2:
                return self._send_json(HTTPStatus.OK, preview.status())
            if segments[2] in PREVIEW_FILES:
                path = preview.file(segments[2])
                if path is None:
                    return self._send_error_json(HTTPStatus.CONFLICT, "the preview is not ready")
                download = query.get("download", [""])[0] == "1"
                name = f"{Path(source.name).stem}_{segments[2]}" if download else None
                return self._send_file(path, "video/mp4", name)

        self._send_error_json(HTTPStatus.NOT_FOUND, "no such endpoint")

    def _route_miv_post(self, segments: list[str], query: dict[str, list[str]]) -> None:
        if segments == ["upload"]:
            return self._upload(query)

        self._drain_body()
        if len(segments) == 3 and segments[0] == "sources" and segments[2] == "preview":
            source = self.config.library.get(segments[1])
            if source is None:
                return self._send_error_json(HTTPStatus.NOT_FOUND,
                                             "unknown .miv file (moved, changed or deleted?)")
            return self._start_preview(source)
        self._send_error_json(HTTPStatus.NOT_FOUND, "no such endpoint")

    def _upload(self, query: dict[str, list[str]]) -> None:
        length = self._upload_length(query, (".miv",), MAX_MIV_UPLOAD_BYTES, "upload.miv")
        if length is None:
            return
        try:
            source = self.config.library.add_upload(
                Path(query.get("filename", ["upload.miv"])[0]).name, self.rfile, length)
        except ValueError as exc:
            self.close_connection = True
            return self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        print(f"  uploaded {source.name} ({source.size_bytes} bytes)")
        self._start_preview(source)

    def _start_preview(self, source: MivSource) -> None:
        preview = self.config.previews.ensure(source)
        self._send_json(HTTPStatus.ACCEPTED,
                        {"source": source.describe(), "preview": preview.status()})

    def _upload_length(
        self,
        query: dict[str, list[str]],
        suffixes: tuple[str, ...],
        limit: int,
        fallback_name: str,
    ) -> int | None:
        """The body length of an acceptable upload, else None once the refusal is sent.

        Checked before a byte of the body is read, so an unwanted multi-gigabyte
        upload is refused rather than received; the connection then closes,
        because what was not read cannot be skipped past.
        """
        length = int(self.headers.get("Content-Length") or 0)
        filename = Path(query.get("filename", [fallback_name])[0]).name
        if length <= 0:
            refusal = (HTTPStatus.BAD_REQUEST, "empty upload")
        elif length > limit:
            refusal = (HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                       f"file is larger than {limit // 2**30} GB")
        elif not filename.lower().endswith(suffixes):
            refusal = (HTTPStatus.BAD_REQUEST,
                       f"expected {' or '.join(suffixes)}, got {filename!r}")
        else:
            return length
        self.close_connection = True  # the body was not read
        self._send_error_json(*refusal)
        return None

    def _drain_body(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 0:
            self.rfile.read(length)

    # -- /api/video -------------------------------------------------------

    def _route_video_get(self, segments: list[str]) -> None:
        if len(segments) == 2 and segments[0] == "conversions":
            conversion = self.config.conversions.get(segments[1])
            if conversion is None:
                return self._send_error_json(HTTPStatus.NOT_FOUND, "unknown conversion")
            return self._send_json(HTTPStatus.OK, conversion.status())
        self._send_error_json(HTTPStatus.NOT_FOUND, "no such endpoint")

    def _route_video_post(self, segments: list[str], query: dict[str, list[str]]) -> None:
        if segments == ["upload"]:
            return self._convert_upload(query)

        self._drain_body()
        if len(segments) == 3 and segments[0] == "conversions" and segments[2] == "cancel":
            conversion = self.config.conversions.cancel(segments[1])
            if conversion is None:
                return self._send_error_json(HTTPStatus.NOT_FOUND, "unknown conversion")
            return self._send_json(HTTPStatus.OK, conversion.status())
        self._send_error_json(HTTPStatus.NOT_FOUND, "no such endpoint")

    def _convert_upload(self, query: dict[str, list[str]]) -> None:
        """Store an uploaded video and queue its conversion to MIV.

        The options are parsed before the body is read, so a typo in them costs
        the browser a rejected request rather than a finished upload.
        """
        try:
            options = ConversionOptions.from_query(query)
        except ValueError as exc:
            self.close_connection = True
            return self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        length = self._upload_length(query, VIDEO_SUFFIXES, MAX_VIDEO_UPLOAD_BYTES, "upload.mp4")
        if length is None:
            return
        try:
            path = self.config.conversions.add_upload(
                Path(query.get("filename", ["upload.mp4"])[0]).name, self.rfile, length)
        except ValueError as exc:
            self.close_connection = True
            return self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        conversion = self.config.conversions.start(path, options)
        print(f"  converting {conversion.name} ({length} bytes) -> {conversion.output_path}")
        self._send_json(HTTPStatus.ACCEPTED, {"conversion": conversion.status()})


class MediaServerConfig(ServerConfig):
    """Stage 1's settings plus the .miv library, the preview store and the converter."""

    def __init__(self, *args: Any, library: Library, previews: PreviewStore,
                 conversions: ConversionStore | None = None, **kwargs: Any):
        """Hold Stage 1's settings (`*args`, `**kwargs`) plus the video side's state."""
        super().__init__(*args, **kwargs)
        self.library = library
        self.previews = previews
        self.conversions = conversions or ConversionStore(
            previews.cache_dir, previews.tmiv, previews.tmiv_error)

    def describe_library(self) -> dict[str, Any]:
        """The video page's startup and refresh payload."""
        sources = []
        for source in self.library.sources():
            preview = self.previews.get(source)
            sources.append(dict(source.describe(),
                                preview=preview.status(include_meta=False) if preview else None))
        tmiv = self.previews.tmiv
        return {
            "sources": sources,
            "roots": [str(root) for root in self.library.roots],
            "tmiv": {
                "ok": tmiv is not None,
                "version": tmiv.marker.get("version") if tmiv else None,
                "root": str(tmiv.root) if tmiv else None,
                "error": self.previews.tmiv_error,
            },
            "cache_dir": str(self.previews.cache_dir),
            "max_upload_mb": MAX_MIV_UPLOAD_BYTES // 2**20,
            "conversions": [conversion.status() for conversion in self.conversions.recent()],
            "convert": self.describe_converter(),
        }

    def describe_converter(self) -> dict[str, Any]:
        """What the browser needs to offer a conversion: what it accepts, and the defaults."""
        return {
            "suffixes": list(VIDEO_SUFFIXES),
            "max_upload_mb": MAX_VIDEO_UPLOAD_BYTES // 2**20,
            "view_heights": list(VIEW_HEIGHTS),
            "min_angle_deg": SLIDER_MIN_ANGLE_DEG,
            "max_angle_deg": MAX_ANGLE_DEG,
            "defaults": ConversionOptions().describe(),
            "device": self.conversions.device,
            "precision": self.conversions.precision,
        }


def resolve_tmiv(explicit: Path | None = None) -> tuple[TmivInstall | None, str | None]:
    """The TMIV build to decode with, or why there is none.

    Looks where the Stage 2 tools do (`--tmiv-dir`, `$SHARP_TMIV_DIR`, `./.tmiv`),
    then in this checkout's `.tmiv`, so the server can be started from any folder.
    """
    try:
        return find_tmiv(explicit), None
    except TmivNotFoundError as exc:
        if explicit is None and (REPO_ROOT / ".tmiv").is_dir():
            try:
                return find_tmiv(REPO_ROOT / ".tmiv"), None
            except TmivNotFoundError as repo_exc:
                return None, str(repo_exc)
        return None, str(exc)


def default_cache_dir() -> Path:
    """`$XDG_CACHE_HOME/sharp-video-webui`, else `~/.cache/sharp-video-webui`."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "sharp-video-webui"


def serve(
    paths: list[Path],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    cache_dir: Path | None = None,
    tmiv_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    device: str = "default",
    precision: str = "fp16",
    open_browser: bool = False,
) -> None:
    """Run the web UI until interrupted."""
    cache_dir = cache_dir or default_cache_dir()
    tmiv, tmiv_error = resolve_tmiv(tmiv_dir)
    conversions = ConversionStore(cache_dir, tmiv, tmiv_error, checkpoint_path=checkpoint_path,
                                  device=device, precision=precision)
    library = Library(paths, cache_dir / "uploads", conversions.converted_dir)
    previews = PreviewStore(cache_dir, tmiv, tmiv_error)
    store = JobStore()
    sample = photo_server.SAMPLE_IMAGE if photo_server.SAMPLE_IMAGE.is_file() else None
    config = MediaServerConfig(store, checkpoint_path, device, precision, sample,
                               library=library, previews=previews, conversions=conversions)
    handler = partial(SpatialMediaHandler, config=config)

    with ThreadingHTTPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/video"
        print(f"sharp-video-webui -> {url}   (spatial photos: http://{host}:{port}/)")
        print(f"  library: {', '.join(str(root) for root in library.roots)}")
        print(f"  TMIV:    {tmiv.root if tmiv else 'not found -- ' + str(tmiv_error)}")
        print(f"  convert: video -> MIV on {device} ({precision}); clips land in "
              f"{conversions.converted_dir}")
        print(f"  cache:   {cache_dir}")
        print("  Ctrl-C to stop.")
        if open_browser:
            threading.Timer(0.5, webbrowser.open, args=(url,)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down.")
        finally:
            store.clear()


def main(argv: list[str] | None = None) -> None:
    """Entry point for `sharp-video-webui` and `python -m sharp_video.webui`."""
    parser = argparse.ArgumentParser(
        description="Convert videos to MIV and preview spatial media in a browser.")
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help=".miv files, or folders whose *.miv files to list (default: the current folder). "
             "Videos converted here are added to the library wherever it points.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default: %(default)s).")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port (default: %(default)s)."
    )
    parser.add_argument(
        "--tmiv-dir", type=Path, default=None,
        help="TMIV build (default: $SHARP_TMIV_DIR, ./.tmiv, then this checkout's .tmiv).",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Where previews and uploads are kept (default: ~/.cache/sharp-video-webui).",
    )
    parser.add_argument(
        "-c", "--checkpoint", type=Path, default=None,
        help="SHARP checkpoint for the photo page and video conversions (downloads the "
             "default model if omitted).",
    )
    parser.add_argument(
        "--device", default="default",
        help="Stage 1 device: cpu, mps, cuda, or default (auto-detect). Converting a video "
             "renders 9 views, which needs CUDA.",
    )
    parser.add_argument(
        "--precision", choices=["fp32", "fp16", "bf16"], default="fp16",
        help="Stage 1 forward-pass precision (default: %(default)s).",
    )
    parser.add_argument("--open", action="store_true", help="Open a browser on startup.")
    args = parser.parse_args(argv)

    serve(
        paths=args.paths or [Path.cwd()],
        host=args.host,
        port=args.port,
        cache_dir=args.cache_dir,
        tmiv_dir=args.tmiv_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
        precision=args.precision,
        open_browser=args.open,
    )
