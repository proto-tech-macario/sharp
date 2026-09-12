"""Standalone MIV encoder: SpatialSequence -> MIV (spec §22-§26).

Consumes only the `sharp_video.contract` types (RGB + depth + camera +
timing). It never imports SHARP, torch, Gaussian code, or checkpoints, so the
spatial-authoring technology upstream can be replaced without touching it.
"""
