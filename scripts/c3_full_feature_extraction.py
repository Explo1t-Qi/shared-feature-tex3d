from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import _full_feature_extraction_common as common  # noqa: E402


CHECKPOINT_IDENTITY = "gs://openpi-assets/checkpoints/pi05_libero"
CONFIG_NAME = "pi05_libero"
EXTRACTION_IDENTITY = "pi05_libero:PI0Pytorch"
DEFAULT_OPENPI_ROOT = PROJECT_ROOT.parent / "openpi"
DEFAULT_CHECKPOINT_DIR = Path("/data/xiaomengqi/checkpoints/pi05_libero_pytorch")
SPEC = common.FeatureSpec(
    model_family="pi05",
    source_model="pi05",
    checkpoint_identity=CHECKPOINT_IDENTITY,
    feature_schema_version="pi05_torch_features_v1",
    manifest_filename="pi05_feature_manifest.json",
    nodes=(
        common.FeatureNode("P1", "p1_siglip", (256, 1152)),
        common.FeatureNode("P2", "p2_projected", (256, 2048)),
    ),
    feature_config=EXTRACTION_IDENTITY,
)


@dataclass(frozen=True)
class _Runtime:
    model: Any
    policy: Any
    extractor: Callable[..., Sequence[Path]]


@dataclass(frozen=True)
class _OpenPIComponents:
    torch: Any
    model_type: Any
    policy_config: Any
    config: Any
    extractor: Callable[..., Sequence[Path]]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Pilot v0.2 feature extraction with current PI0Pytorch."
    )
    parser.add_argument(
        "--collection-manifest",
        type=Path,
        required=True,
        help="Completed Pilot v0.2 collection_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh or resumable Phase 1 pi0.5 feature directory.",
    )
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=DEFAULT_OPENPI_ROOT,
        help="OpenPI source repository containing the current PI0Pytorch code.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Current pi05_libero PyTorch checkpoint directory.",
    )
    return parser.parse_args(argv)


def _validate_openpi_root(path: str | Path) -> Path:
    openpi_root = Path(path).expanduser().resolve()
    if not openpi_root.is_dir():
        raise FileNotFoundError(f"OpenPI repository not found: {openpi_root}")
    if not (openpi_root / "src" / "openpi").is_dir():
        raise FileNotFoundError(
            f"OpenPI Python source package not found under: {openpi_root}"
        )
    if not (
        openpi_root / "packages" / "openpi-client" / "src" / "openpi_client"
    ).is_dir():
        raise FileNotFoundError(
            f"OpenPI client source package not found under: {openpi_root}"
        )
    return openpi_root


def _validate_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"PI0Pytorch model.safetensors not found under: {checkpoint}"
        )
    if not (checkpoint / "assets").is_dir():
        raise FileNotFoundError(
            f"PI0Pytorch checkpoint assets not found under: {checkpoint}"
        )
    return checkpoint


def _validate_train_config(train_config: Any) -> None:
    model = getattr(train_config, "model", None)
    if (
        getattr(train_config, "name", None) != CONFIG_NAME
        or model is None
        or getattr(model, "pi05", None) is not True
        or getattr(model, "action_horizon", None) != 10
        or getattr(model, "action_dim", None) != 32
        or getattr(model, "discrete_state_input", None) is not False
        or getattr(model, "max_token_len", None) != 200
    ):
        raise RuntimeError("pi05_libero TrainConfig violates frozen semantics")


def _load_openpi_components(openpi_root: Path) -> _OpenPIComponents:
    source_roots = (
        PROJECT_ROOT,
        openpi_root / "packages" / "openpi-client" / "src",
        openpi_root / "src",
    )
    for source_root in source_roots:
        source_string = str(source_root)
        if source_string not in sys.path:
            sys.path.insert(0, source_string)

    import torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies import policy_config
    from openpi.training import config
    from shared_feature import extract_pi05_features

    return _OpenPIComponents(
        torch=torch,
        model_type=PI0Pytorch,
        policy_config=policy_config,
        config=config,
        extractor=extract_pi05_features,
    )


def _load_runtime(openpi_root: Path, checkpoint: Path) -> _Runtime:
    components = _load_openpi_components(openpi_root)
    if not components.torch.cuda.is_available():
        raise RuntimeError("PI0Pytorch feature extraction requires a CUDA device")
    train_config = components.config.get_config(CONFIG_NAME)
    _validate_train_config(train_config)
    policy = components.policy_config.create_trained_policy(
        train_config,
        checkpoint,
        pytorch_device="cuda",
    )
    if getattr(policy, "_is_pytorch_model", None) is not True:
        raise RuntimeError("current extraction did not load a PI0Pytorch policy")
    model = getattr(policy, "_model", None)
    if not isinstance(model, components.model_type):
        raise RuntimeError("pi0.5 policy model is not the current PI0Pytorch class")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("PI0Pytorch model was not frozen in eval mode")
    return _Runtime(
        model=model,
        policy=policy,
        extractor=components.extractor,
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source = common.load_source_collection(args.collection_manifest)
    openpi_root = _validate_openpi_root(args.openpi_root)
    checkpoint = _validate_checkpoint(args.checkpoint_dir)
    preparation = common.prepare_output(
        output_dir=args.output_dir,
        source=source,
        spec=SPEC,
    )

    extracted_count = 0
    if preparation.missing_records:
        runtime = _load_runtime(openpi_root, checkpoint)
        returned_paths = runtime.extractor(
            model=runtime.model,
            policy=runtime.policy,
            checkpoint=CHECKPOINT_IDENTITY,
            observation_paths=tuple(
                record.resolved_observation_path
                for record in preparation.missing_records
            ),
            output_dir=preparation.features_dir,
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
        "status": "Phase 1 PI0Pytorch Feature Extraction — COMPLETE",
        "manifest_path": str(preparation.manifest_path),
        "num_feature_archives": len(feature_paths),
        "num_node_tensors": len(feature_paths) * len(SPEC.nodes),
        "reused_archives": preparation.reused_count,
        "extracted_archives": extracted_count,
        "backend": "PI0Pytorch",
        "checkpoint_path": str(checkpoint),
    }


def main(argv: Sequence[str] | None = None) -> int:
    summary = _run(_parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
