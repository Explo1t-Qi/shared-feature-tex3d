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
DEFAULT_OPENPI_ROOT = PROJECT_ROOT.parent / "openpi"
SPEC = common.FeatureSpec(
    model_family="pi05",
    source_model="pi05",
    checkpoint_identity=CHECKPOINT_IDENTITY,
    feature_schema_version="pi05_features_v1",
    manifest_filename="pi05_feature_manifest.json",
    nodes=(
        common.FeatureNode("P1", "p1_siglip", (256, 1152)),
        common.FeatureNode("P2", "p2_projected", (256, 2048)),
    ),
    feature_config=CONFIG_NAME,
)


@dataclass(frozen=True)
class _Runtime:
    model: Any
    train_config: Any
    norm_stats: Any
    extractor: Callable[..., Sequence[Path]]


@dataclass(frozen=True)
class _OpenPIComponents:
    jax: Any
    jnp: Any
    openpi_model: Any
    download: Any
    checkpoints: Any
    config: Any
    extractor: Callable[..., Sequence[Path]]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal Pilot v0.2 pi0.5 feature extraction."
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
        help="Fresh or resumable formal C3 output directory.",
    )
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=DEFAULT_OPENPI_ROOT,
        help="Local OpenPI source repository used by the validated C3 path.",
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


def _load_model(train_config: Any, checkpoint_path: Path, openpi_model: Any, jnp: Any):
    safetensors_path = checkpoint_path / "model.safetensors"
    if safetensors_path.is_file():
        model = train_config.model.load_pytorch(
            train_config,
            str(safetensors_path),
        )
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        return model, "pytorch"

    params_path = checkpoint_path / "params"
    if not params_path.is_dir():
        raise FileNotFoundError(
            "pi0.5 checkpoint contains neither model.safetensors nor params"
        )
    params = openpi_model.restore_params(params_path, dtype=jnp.bfloat16)
    return train_config.model.load(params), "jax_nnx"


def _validate_runtime_precision(model: Any, backend: str, jax: Any) -> None:
    try:
        if backend == "jax_nnx":
            from flax import nnx

            leaves = jax.tree.leaves(nnx.state(model, nnx.Param))
            parameter = next(
                value
                for leaf in leaves
                if hasattr((value := getattr(leaf, "value", leaf)), "dtype")
            )
        else:
            parameter = next(model.parameters())
    except (AttributeError, StopIteration, TypeError) as error:
        raise RuntimeError("cannot inspect pi0.5 model precision") from error
    if str(parameter.dtype) not in {"bfloat16", "torch.bfloat16"}:
        raise RuntimeError(f"pi0.5 model is not BF16: {parameter.dtype}")


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

    import jax
    import jax.numpy as jnp
    from openpi.models import model as openpi_model
    from openpi.shared import download
    from openpi.training import checkpoints, config
    from shared_feature import extract_pi05_features

    return _OpenPIComponents(
        jax=jax,
        jnp=jnp,
        openpi_model=openpi_model,
        download=download,
        checkpoints=checkpoints,
        config=config,
        extractor=extract_pi05_features,
    )


def _load_runtime(openpi_root: Path) -> _Runtime:
    components = _load_openpi_components(openpi_root)
    if components.jax.default_backend() != "gpu":
        raise RuntimeError(
            "pi0.5 extraction requires JAX GPU backend, got "
            f"{components.jax.default_backend()}"
        )
    train_config = components.config.get_config(CONFIG_NAME)
    checkpoint_path = components.download.maybe_download(CHECKPOINT_IDENTITY).resolve()
    data_config = train_config.data.create(
        train_config.assets_dirs,
        train_config.model,
    )
    norm_stats = components.checkpoints.load_norm_stats(
        checkpoint_path / "assets",
        data_config.asset_id,
    )
    model, backend = _load_model(
        train_config,
        checkpoint_path,
        components.openpi_model,
        components.jnp,
    )
    image_encoder = getattr(getattr(model, "PaliGemma", None), "img", None)
    if not callable(image_encoder):
        raise RuntimeError("pi0.5 model does not expose callable PaliGemma.img")
    _validate_runtime_precision(model, backend, components.jax)
    return _Runtime(
        model=model,
        train_config=train_config,
        norm_stats=norm_stats,
        extractor=components.extractor,
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source = common.load_source_collection(args.collection_manifest)
    openpi_root = _validate_openpi_root(args.openpi_root)
    preparation = common.prepare_output(
        output_dir=args.output_dir,
        source=source,
        spec=SPEC,
    )

    extracted_count = 0
    if preparation.missing_records:
        runtime = _load_runtime(openpi_root)
        returned_paths = runtime.extractor(
            model=runtime.model,
            train_config=runtime.train_config,
            checkpoint=CHECKPOINT_IDENTITY,
            norm_stats=runtime.norm_stats,
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
        "status": "C3 pi0.5 Full Feature Extraction — COMPLETE",
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
