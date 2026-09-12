"""TMIV configuration for coding a 9-view spatial sequence as MIV.

Choices (see docs/stage2_miv.md):
- `NoViewOptimizer` + `NoPruner`: all 9 views are coded in full, so the MIV
  file carries the complete multi-view RGB + depth representation and the
  decoder reconstructs exactly v0..v8.
- `interPeriod = 1`: every frame is its own access unit with its own common
  atlas frame, which is where per-frame camera poses (MIV view parameter
  updates) are carried.
- Per-frame camera files `S2/seq/NNNNNN.json`, read by the patched encoder.
- Geometry at full resolution (`geometryScaleEnabledFlag = false`), occupancy
  embedded in geometry, texture and geometry video 10-bit VVC.

All path formats are relative to TMIV's input/output directories.
"""

from __future__ import annotations

import math

CONTENT_ID = "S2"
RATE_ID = "RP1"
VIEW_NAMES = tuple(f"v{i}" for i in range(9))

TEXTURE_BIT_DEPTH = 10
GEOMETRY_BIT_DEPTH = 16
TEXTURE_FORMAT = "yuv420p10le"
GEOMETRY_FORMAT = "yuv420p16le"

BLOCK_SIZE = 16
MAX_LUMA_PICTURE_SIZE = 8912896  # per atlas, as in TMIV's CTC anchors
PACKING_MARGIN = 1.25


def texture_input_path(view_name: str, width: int, height: int) -> str:
    """Path of a view's texture file, relative to TMIV's input directory."""
    return f"{CONTENT_ID}/{view_name}_texture_{width}x{height}_{TEXTURE_FORMAT}.yuv"


def geometry_input_path(view_name: str, width: int, height: int) -> str:
    """Path of a view's depth (geometry) file, relative to TMIV's input directory."""
    return f"{CONTENT_ID}/{view_name}_depth_{width}x{height}_{GEOMETRY_FORMAT}.yuv"


def sequence_config_path(frame_index: int) -> str:
    """Path of frame `frame_index`'s camera file, relative to TMIV's input directory."""
    return f"{CONTENT_ID}/seq/{frame_index:06d}.json"


def bitstream_output_path() -> str:
    """Where the multiplexer writes the final MIV bitstream (relative to TMIV's output dir)."""
    return multiplexer_config()["outputBitstreamPathFmt"].format(0, CONTENT_ID, RATE_ID)


def sequence_config(
    cameras: list[dict], fps: float, frame_count: int,
    content_name: str = "SHARP Stage 2 spatial video",
) -> dict:
    """A TMIV sequence configuration (format version 4) for one frame's cameras."""
    return {
        "Version": "4.0",
        "Content_name": content_name,
        "BoundingBox_center": [0.0, 0.0, 0.0],
        "Fps": float(fps),
        "Frames_number": int(frame_count),
        "lengthsInMeters": True,
        "sourceCameraNames": [camera["Name"] for camera in cameras],
        "cameras": cameras,
    }


def atlas_budget(width: int, height: int, fps: float, views: int = 9) -> tuple[int, int, float]:
    """(maxAtlases, maxLumaPictureSize, maxLumaSampleRate) that fit every view in full.

    TMIV derives the atlas count and size from the luma sample-rate budget
    (texture + full-resolution geometry per atlas sample) and the per-atlas
    picture limit; this sizes both so the 9 complete views always fit.
    """
    padded = (math.ceil(width / BLOCK_SIZE) * BLOCK_SIZE) * (math.ceil(height / BLOCK_SIZE)
                                                             * BLOCK_SIZE)
    total = int(views * padded * PACKING_MARGIN)
    block_area = BLOCK_SIZE * BLOCK_SIZE
    picture_size = min(MAX_LUMA_PICTURE_SIZE, math.ceil(total / block_area) * block_area)
    max_atlases = math.ceil(total / picture_size)
    samples_per_atlas_sample = 2.0  # texture + geometry, both at full resolution
    sample_rate = float(max_atlases * picture_size * samples_per_atlas_sample * fps)
    return max_atlases, picture_size, sample_rate


def encoder_config(width: int, height: int, fps: float, intra_period: int = 32) -> dict:
    """TmivEncoder configuration for coding all 9 views with per-frame cameras."""
    max_atlases, picture_size, sample_rate = atlas_budget(width, height, fps)
    return {
        "Aggregator": {},
        "AggregatorMethod": "Aggregator",
        "depthLowQualityFlag": False,  # rendered depth: consistent by construction
        "NoPruner": {},
        "PrunerMethod": "NoPruner",
        "NoViewOptimizer": {},
        "ViewOptimizerMethod": "NoViewOptimizer",
        "Packer": {
            "enableMerging": True,
            "enablePatchInPatch": True,
            "enablePatchInformation": False,
            "enableRecursiveSplit": True,
            "minPatchSize": 16,
            "overlap": 1,
            "sortingMethod": 0,
        },
        "PackerMethod": "Packer",
        "bitDepthGeometryVideo": 10,
        "bitDepthTextureVideo": TEXTURE_BIT_DEPTH,
        "blockSize": BLOCK_SIZE,
        "chromaScaleEnabledFlag": False,
        "codecGroupIdc": "VVC Main10",
        "colorizedGeometryEnabledFlag": False,
        "configDirectory": ".",
        "depthOccThresholdAsymmetry": 1.5,
        "depthOccThresholdIfSet": [0.00390625, 0.0625],
        "dynamicDepthRange": True,
        "embeddedOccupancy": True,
        "framePacking": False,
        "geometryScaleEnabledFlag": False,
        "halveDepthRange": False,
        "haveGeometryVideo": True,
        "haveOccupancyVideo": False,
        "haveTextureVideo": True,
        "informationPruning": False,
        "inputDirectory": ".",
        "inputGeometryPathFmt": "{1}/{3}_depth_{4}x{5}_{6}.yuv",
        "inputSequenceConfigPathFmt": "{1}/seq/{3:06}.json",
        "inputTexturePathFmt": "{1}/{3}_texture_{4}x{5}_{6}.yuv",
        "interPeriod": 1,
        "intraPeriod": int(intra_period),
        "levelIdc": "8.5",
        "maxAtlases": max_atlases,
        "maxEntityId": 0,
        "maxLumaPictureSize": picture_size,
        "maxLumaSampleRate": sample_rate,
        "numGroups": 1,
        "oneV3cFrameOnly": False,
        "oneViewPerAtlasFlag": False,
        "outputBitstreamPathFmt": "{1}/RP0/TMIV_{1}_RP0.bit",
        "outputDirectory": ".",
        "outputGeometryVideoDataPathFmt": "{1}/RP0/TMIV_{1}_RP0_geo_c{3:02}_{4}x{5}_{6}.yuv",
        "outputTextureVideoDataPathFmt": "{1}/RP0/TMIV_{1}_RP0_tex_c{3:02}_{4}x{5}_{6}.yuv",
        "patchMarginEnabledFlag": False,
        "patchRedundancyRemoval": False,
        "piecewiseDepthLinearScaling": False,
        "reconstructionIdc": "Rec Unconstrained",
        "rewriteParameterSets": False,
        "textureOffsetEnabledFlag": False,
        "toolsetIdc": "MIV 2",
        "viewportCameraParametersSei": False,
        "viewportPositionSei": False,
    }


def multiplexer_config() -> dict:
    """TmivMultiplexer configuration: MIV metadata + VVC sub-bitstreams -> one file."""
    return {
        "inputBitstreamPathFmt": "{1}/RP0/TMIV_{1}_RP0.bit",
        "inputDirectory": ".",
        "inputGeometryVideoSubBitstreamPathFmt": "{1}/{2}/TMIV_{1}_{2}_geo_c{3:02}.bit",
        "inputTextureVideoSubBitstreamPathFmt": "{1}/{2}/TMIV_{1}_{2}_tex_c{3:02}.bit",
        "outputBitstreamPathFmt": "{1}/{2}/TMIV_{1}_{2}.bit",
        "outputDirectory": ".",
    }


def _escape_fmt(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def decoder_config(bitstream_path) -> dict:
    """TmivDecoder configuration that reconstructs every view and writes per-frame cameras."""
    return {
        "configDirectory": ".",
        "inputBitstreamPathFmt": _escape_fmt(str(bitstream_path)),
        "inputDirectory": ".",
        "outputDirectory": ".",
        "outputMultiviewGeometryPathFmt": "decoded/{3:02}_geo_{4}x{5}_{6}.yuv",
        "outputMultiviewOccupancyPathFmt": "decoded/{3:02}_occ_{4}x{5}_{6}.yuv",
        "outputMultiviewTexturePathFmt": "decoded/{3:02}_tex_{4}x{5}_{6}.yuv",
        "outputSequenceConfigPathFmt": "decoded/seq_{3:06}.json",
    }


def qp_csv(qp_texture: int, qp_geometry: int) -> str:
    """Quantization parameters in the rate-table format TMIV's encode.py reads."""
    return (
        f"component_id,rates,{RATE_ID}\n"
        f"geo,,{int(qp_geometry)}\n"
        f"tex,,{int(qp_texture)}\n"
    )
