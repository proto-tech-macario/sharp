"""A local web UI for spatial media: Stage 1's photo page plus MIV video previews.

`sharp-video-webui [FILES_OR_FOLDERS...]` (or `python -m sharp_video.webui`)
serves Stage 1's spatial-photo page at `/` and a video page at `/video` that
decodes MIV files with TmivDecoder and plays their 9 views with parallax.

Standard library + numpy/Pillow/PyAV only -- no web framework.
"""
