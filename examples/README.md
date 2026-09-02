# Examples

`../data/teaser.jpg` is the bundled sample input (also used by `webui/`).

Generating `spatial_photo.h5` requires a CUDA GPU (the renderer uses SHARP's
gsplat-based rasterizer, which has no CPU/MPS kernel), so this repo does not
ship a pre-generated output file. To produce one on a CUDA machine:

```bash
sharp-spatialize --input ../data/teaser.jpg --angle 10 --output spatial_photo.h5
sharp-spatialize-validate spatial_photo.h5
```

See `docs/spatial_photo_format.md` for the file format.
