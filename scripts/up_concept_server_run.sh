#!/usr/bin/env bash
# 用法：bash scripts/up_concept_server_run.sh <完整 commit SHA> [--preflight-only|--run]
# --run 需要显式 GPU_ID；不自动同步 Git、不安装环境、不重跑已有目录。
set -euo pipefail

EXPECTED_HEAD="${1:?Pass the exact committed project SHA}"
MODE="${2:---preflight-only}"
case "$MODE" in --preflight-only|--run) ;; *) echo "Invalid mode: $MODE" >&2; exit 2 ;; esac
if [[ ! "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]; then
    echo "EXPECTED_HEAD must be a full 40-character SHA" >&2
    exit 2
fi

UP_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OPENVLA_PY="${OPENVLA_PY:-/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python}"
TEX3D_ROOT="${TEX3D_ROOT:-/data/xiaomengqi/src/tex3d-fixed}"
OPENVLA_CKPT="${OPENVLA_CKPT:-/data/huangsimin/openvla-7b-finetuned-libero-spatial}"
COLLECTION_MANIFEST="${COLLECTION_MANIFEST:-$UP_REPO/experiment_inbox/c5_d0_pilot_v02_full_collection/collection_manifest.json}"
UP_OUTPUT_ROOT="${UP_OUTPUT_ROOT:-/data/xiaomengqi/logs/up-concept}"
UP_RUN_ID="${UP_RUN_ID:-up-concept-v1-${EXPECTED_HEAD:0:12}}"
if [[ ! "$UP_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "UP_RUN_ID must be a simple directory name" >&2
    exit 2
fi
UP_RUN_DIR="$UP_OUTPUT_ROOT/$UP_RUN_ID"
[[ -x "$OPENVLA_PY" ]] || { echo "Python not executable: $OPENVLA_PY" >&2; exit 1; }
[[ -d "$TEX3D_ROOT/openvla" ]] || { echo "Missing Tex3D root: $TEX3D_ROOT" >&2; exit 1; }
[[ -d "$OPENVLA_CKPT" ]] || { echo "Missing checkpoint: $OPENVLA_CKPT" >&2; exit 1; }
[[ -f "$COLLECTION_MANIFEST" ]] || { echo "Missing collection: $COLLECTION_MANIFEST" >&2; exit 1; }
[[ ! -e "$UP_RUN_DIR" ]] || { echo "Output exists; refusing resume: $UP_RUN_DIR" >&2; exit 1; }
[[ "$(git -C "$UP_REPO" rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || {
    echo "Project HEAD does not match $EXPECTED_HEAD" >&2; exit 1;
}
git -C "$UP_REPO" diff --exit-code HEAD -- >/dev/null || {
    echo "Project has tracked modifications" >&2; exit 1;
}
# 历史 experiment_inbox 等未跟踪产物允许存在；未提交的新执行文件不允许。
for up_file in shared_feature/up_concept.py scripts/up_concept_pilot.py scripts/up_concept_server_run.sh; do
    git -C "$UP_REPO" ls-files --error-unmatch "$up_file" >/dev/null
done

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# 只使用当前项目和已指定 Tex3D 的 Python 代码；其余依赖使用所选 Python 环境。
export PYTHONPATH="$UP_REPO:$TEX3D_ROOT/openvla"
ARGS=(--collection-manifest "$COLLECTION_MANIFEST" --pretrained-checkpoint "$OPENVLA_CKPT"
      --tex3d-openvla-root "$TEX3D_ROOT/openvla" --output-dir "$UP_RUN_DIR"
      --expected-head "$EXPECTED_HEAD")

CUDA_VISIBLE_DEVICES='' "$OPENVLA_PY" "$UP_REPO/scripts/up_concept_pilot.py" "${ARGS[@]}" --preflight-only
if [[ "$MODE" == --preflight-only ]]; then
    exit 0
fi
: "${GPU_ID:?Set GPU_ID to one available physical GPU before --run}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be one physical GPU number" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU_ID"
mkdir -p -- "$UP_OUTPUT_ROOT"
UP_CONSOLE=$(mktemp "$UP_OUTPUT_ROOT/$UP_RUN_ID.console.XXXXXX.log")
exec > >(tee "$UP_CONSOLE") 2>&1
trap 'echo "FAILED at line $LINENO; retain partial outputs and console log: $UP_CONSOLE" >&2' ERR
printf 'GPU_ID=%s\nHEAD=%s\nCONSOLE=%s\nOUTPUT=%s\n' "$GPU_ID" "$EXPECTED_HEAD" "$UP_CONSOLE" "$UP_RUN_DIR"

CUDA_VISIBLE_DEVICES='' "$OPENVLA_PY" -m pytest -q -p no:cacheprovider \
    "$UP_REPO/tests/test_up_concept.py" "$UP_REPO/tests/test_up_concept_runner.py"
"$OPENVLA_PY" "$UP_REPO/scripts/up_concept_pilot.py" "${ARGS[@]}"
test -f "$UP_RUN_DIR/results.json"
test ! -e "$UP_RUN_DIR/failure.json"
UP_BUNDLE="$UP_RUN_DIR.review.tar.gz"
test ! -e "$UP_BUNDLE"
# logits 保留服务器；只打包可直接审阅的候选、校准、配对结果与摘要。
tar -czf "$UP_BUNDLE" -C "$UP_RUN_DIR" \
    protocol.json checkpoint_hashes.json candidates.json selected.json calibration.json \
    paired_results.json results.json summary.csv summary.md
printf 'UP PILOT SERVER FINISHED\nReview bundle: %s\nFull logits: %s/logits\n' "$UP_BUNDLE" "$UP_RUN_DIR"
