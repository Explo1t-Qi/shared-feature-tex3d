from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import _full_feature_extraction_common as common  # noqa: E402


CHECKPOINT_IDENTITY = "openvla/openvla-7b-finetuned-libero-spatial"
UNNORM_KEY = "libero_spatial_no_noops"
GLOBAL_SEED = 7
DEFAULT_TEX3D_OPENVLA_ROOT = PROJECT_ROOT.parent / "tex3d" / "openvla"
SPEC = common.FeatureSpec(
    model_family="openvla",
    source_model="openvla",
    checkpoint_identity=CHECKPOINT_IDENTITY,
    feature_schema_version="openvla_features_v1",
    manifest_filename="openvla_feature_manifest.json",
    nodes=(
        common.FeatureNode("O1-S", "o1_siglip", (256, 1152)),
        common.FeatureNode("O1-F", "o1_fused", (256, 2176)),
        common.FeatureNode("O2", "o2_projected", (256, 4096)),
    ),
)


@dataclass(frozen=True)
class _Runtime:
    torch: Any
    get_model: Callable[[Any], Any]
    get_processor: Callable[[Any], Any]
    set_seed_everywhere: Callable[[int], None]
    extractor: Callable[..., Sequence[Path]]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal Pilot v0.2 OpenVLA feature extraction."
    )
    parser.add_argument(
        "--collection-manifest",
        type=Path,
        required=True,
        help="Completed Pilot v0.2 collection_manifest.json.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        required=True,
        help="Local OpenVLA LIBERO-Spatial checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh or resumable formal C2 output directory.",
    )
    parser.add_argument(
        "--tex3d-openvla-root",
        type=Path,
        default=DEFAULT_TEX3D_OPENVLA_ROOT,
        help="Official Tex3D OpenVLA integration source root.",
    )
    return parser.parse_args(argv)


def _validate_runtime_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint = args.pretrained_checkpoint.expanduser().resolve()
    tex3d_root = args.tex3d_openvla_root.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint}")
    statistics_path = checkpoint / "dataset_statistics.json"
    if not statistics_path.is_file():
        raise FileNotFoundError(f"dataset statistics file not found: {statistics_path}")
    try:
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(
            f"failed to load dataset statistics: {statistics_path}"
        ) from error
    if not isinstance(statistics, dict) or UNNORM_KEY not in statistics:
        available = sorted(statistics) if isinstance(statistics, dict) else []
        raise KeyError(
            f"{UNNORM_KEY!r} missing from {statistics_path}; available={available}"
        )
    if not tex3d_root.is_dir():
        raise FileNotFoundError(f"official Tex3D OpenVLA root not found: {tex3d_root}")
    return checkpoint, tex3d_root


def _load_runtime(tex3d_root: Path) -> _Runtime:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for source_root in (PROJECT_ROOT, tex3d_root):
        source_string = str(source_root)
        if source_string not in sys.path:
            sys.path.insert(0, source_string)

    import torch
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model, set_seed_everywhere
    from shared_feature import extract_openvla_features

    return _Runtime(
        torch=torch,
        get_model=get_model,
        get_processor=get_processor,
        set_seed_everywhere=set_seed_everywhere,
        extractor=extract_openvla_features,
    )


def _validate_model_runtime(model: Any, torch: Any) -> None:
    vision_backbone = getattr(model, "vision_backbone", None)
    try:
        parameter = next(
            value
            for value in vision_backbone.parameters()
            if torch.is_floating_point(value)
        )
    except (AttributeError, StopIteration) as error:
        raise RuntimeError(
            "OpenVLA vision backbone exposes no floating-point parameter"
        ) from error
    if parameter.device.type != "cuda":
        raise RuntimeError(
            f"OpenVLA vision backbone is not on CUDA: {parameter.device}"
        )
    if parameter.dtype != torch.bfloat16:
        raise RuntimeError(f"OpenVLA vision backbone is not BF16: {parameter.dtype}")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source = common.load_source_collection(args.collection_manifest)
    if source.checkpoint_identity != CHECKPOINT_IDENTITY:
        raise common.FullFeatureExtractionError(
            "C2 feature checkpoint identity differs from source collection"
        )
    checkpoint, tex3d_root = _validate_runtime_paths(args)
    preparation = common.prepare_output(
        output_dir=args.output_dir,
        source=source,
        spec=SPEC,
    )

    extracted_count = 0
    if preparation.missing_records:
        runtime = _load_runtime(tex3d_root)
        if not runtime.torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the C2 extraction runtime")
        runtime.set_seed_everywhere(GLOBAL_SEED)
        model_config = SimpleNamespace(
            model_family="openvla",
            pretrained_checkpoint=str(checkpoint),
            load_in_8bit=False,
            load_in_4bit=False,
            unnorm_key=UNNORM_KEY,
            center_crop=True,
        )
        model = runtime.get_model(model_config)
        processor = runtime.get_processor(model_config)
        _validate_model_runtime(model, runtime.torch)
        returned_paths = runtime.extractor(
            model=model,
            processor=processor,
            pretrained_checkpoint=CHECKPOINT_IDENTITY,
            observation_paths=tuple(
                record.resolved_observation_path
                for record in preparation.missing_records
            ),
            output_dir=preparation.features_dir,
            center_crop=True,
            batch_size=common.BATCH_SIZE,
        )
        common.validate_extractor_paths(
            returned_paths=returned_paths,
            missing_records=preparation.missing_records,
            features_dir=preparation.features_dir,
        )
        extracted_count = len(preparation.missing_records)

    feature_paths = common.validate_complete_output(
        preparation=preparation,
        source=source,
        spec=SPEC,
    )
    manifest = common.build_feature_manifest(source=source, spec=SPEC)
    common.write_manifest_atomic(preparation.manifest_path, manifest)
    return {
        "status": "C2 OpenVLA Full Feature Extraction — COMPLETE",
        "manifest_path": str(preparation.manifest_path),
        "num_feature_archives": len(feature_paths),
        "num_node_tensors": len(feature_paths) * len(SPEC.nodes),
        "reused_archives": preparation.reused_count,
        "extracted_archives": extracted_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    summary = _run(_parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
