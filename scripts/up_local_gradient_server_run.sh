#!/usr/bin/env bash
# bash scripts/up_local_gradient_server_run.sh FULL_SHA [--preflight-only|--run]
set -euo pipefail
EXPECTED_HEAD="${1:?Pass full committed project SHA}"
MODE="${2:---preflight-only}"
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a full SHA' >&2; exit 2; }
case "$MODE" in --preflight-only|--run) ;; *) echo 'Invalid mode' >&2; exit 2 ;; esac
UP_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OPENVLA_PY="${OPENVLA_PY:-/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python}"
TEX3D_ROOT="${TEX3D_ROOT:-/data/xiaomengqi/src/tex3d-fixed}"
OPENVLA_CKPT="${OPENVLA_CKPT:-/data/huangsimin/openvla-7b-finetuned-libero-spatial}"
COLLECTION_MANIFEST="${COLLECTION_MANIFEST:-$UP_REPO/experiment_inbox/c5_d0_pilot_v02_full_collection/collection_manifest.json}"
UP_V1_DIR="${UP_V1_DIR:-/data/xiaomengqi/logs/up-concept/up-concept-v1-b8a330f7d27b}"
UP_SCREEN_DIR="${UP_SCREEN_DIR:-/data/xiaomengqi/logs/up-concept/up-action-screen-75e783b764ef}"
UP_OUTPUT_ROOT="${UP_OUTPUT_ROOT:-/data/xiaomengqi/logs/up-concept}"
UP_RUN_ID="${UP_RUN_ID:-up-local-gradient-${EXPECTED_HEAD:0:12}}"
[[ "$UP_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
UP_RUN_DIR="$UP_OUTPUT_ROOT/$UP_RUN_ID"
[[ -x "$OPENVLA_PY" ]] || { echo "Python missing: $OPENVLA_PY" >&2; exit 1; }
[[ ! -e "$UP_RUN_DIR" ]] || { echo "Refusing existing output: $UP_RUN_DIR" >&2; exit 1; }
[[ "$(git -C "$UP_REPO" rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || { echo 'HEAD mismatch' >&2; exit 1; }
git -C "$UP_REPO" diff --exit-code HEAD -- >/dev/null
for up_file in scripts/up_local_gradient.py shared_feature/up_local_gradient.py scripts/up_local_gradient_server_run.sh tests/test_up_local_gradient.py; do
    git -C "$UP_REPO" ls-files --error-unmatch "$up_file" >/dev/null
done
export PYTHONPATH="$UP_REPO:$TEX3D_ROOT/openvla:$TEX3D_ROOT/openvla/experiments/robot"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false CUDA_DEVICE_ORDER=PCI_BUS_ID TF_CPP_MIN_LOG_LEVEL=3
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
ARGS=(--collection-manifest "$COLLECTION_MANIFEST" --pretrained-checkpoint "$OPENVLA_CKPT"
      --tex3d-openvla-root "$TEX3D_ROOT/openvla" --output-dir "$UP_RUN_DIR"
      --expected-head "$EXPECTED_HEAD" --v1-dir "$UP_V1_DIR" --screen-dir "$UP_SCREEN_DIR")
CUDA_VISIBLE_DEVICES='' "$OPENVLA_PY" "$UP_REPO/scripts/up_local_gradient.py" "${ARGS[@]}" --preflight-only
[[ "$MODE" == --run ]] || exit 0
: "${GPU_ID:?Set one currently available physical GPU}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo 'GPU_ID must be numeric' >&2; exit 2; }
mkdir -p -- "$UP_OUTPUT_ROOT"
UP_CONSOLE=$(mktemp "$UP_OUTPUT_ROOT/$UP_RUN_ID.console.XXXXXX.log")
exec > >(tee "$UP_CONSOLE") 2>&1
trap 'echo "FAILED at line $LINENO; retain $UP_CONSOLE and partial output" >&2' ERR
printf 'LOCAL GRADIENT DIAGNOSTIC\nGPU=%s\nHEAD=%s\nOUTPUT=%s\n' "$GPU_ID" "$EXPECTED_HEAD" "$UP_RUN_DIR"
CUDA_VISIBLE_DEVICES='' "$OPENVLA_PY" -m pytest -q -p no:cacheprovider "$UP_REPO/tests/test_up_local_gradient.py"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$OPENVLA_PY" "$UP_REPO/scripts/up_local_gradient.py" "${ARGS[@]}"
test -f "$UP_RUN_DIR/results.json"
test ! -e "$UP_RUN_DIR/failure.json"
UP_BUNDLE="$UP_RUN_DIR.review.tar.gz"
test ! -e "$UP_BUNDLE"
# 每帧 logits/gradient NPZ 保留服务器；JSON 含全部诊断读出及实际 norm。
tar --exclude='*.npz' -czf "$UP_BUNDLE" -C "$UP_RUN_DIR" .
printf 'Review local diagnostic only: %s\n' "$UP_BUNDLE"
