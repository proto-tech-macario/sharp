"""sharp_video: video -> per-frame Stage 1 spatialization -> temporal HDF5 -> MIV.

Stage 2 of the IRU spatial media pipeline. It wraps the unchanged Stage 1
`sharp_spatialize` package (frames in) and follows it with an MIV encoder
(`sharp_video.miv`, which never imports SHARP, torch, or 3DGS code).

Nothing is imported eagerly here, so `import sharp_video.miv` stays free of
SHARP/torch.
"""
