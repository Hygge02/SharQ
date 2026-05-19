#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL=""
MODE="${MODE:-SHARQ}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PREFILL_SEQ_LEN="${PREFILL_SEQ_LEN:-2048}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
DEVICE="${DEVICE:-cuda:0}"
USE_NSYS="${USE_NSYS:-1}"
OUT=""
TRUST_REMOTE_CODE=0
KV_CACHE=0
NO_EXTRA_FUSION=0
POSITIONAL_MODE=0
PY_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/e2e/run_profile_prefill_nvtx.sh --model /path/to/model [options] [-- extra profile_prefill_nvtx.py args]

Common options:
  --model PATH              HF model path/name. Required.
  --mode MODE               SHARQ, NVFP4, or FP16. Default: SHARQ.
  --seq-len N               Prefill sequence length. Default: 2048.
  --batch-size N            Batch size. Default: 1.
  --warmup-steps N          Warmup prefill runs. Default: 1.
  --device DEVICE           CUDA device. Default: cuda:0.
  -o, --out PATH            Nsight output basename. Default: profiles/<model>_<mode>_prefill_s<seq>_b<batch>.
  --no-conda                Do not activate conda.
  --no-nsys                 Run Python only, without Nsight Systems.
  --trust-remote-code       Pass --trust_remote_code to the Python profiler.
  --kv-cache                Pass --kv_cache to the Python profiler.
  --no-extra-fusion         Pass --no_extra_fusion to the Python profiler.
  -h, --help                Show this message.

Examples:
  bash benchmarks/e2e/run_profile_prefill_nvtx.sh --model /data/Llama-3.1-8B --mode SHARQ --seq-len 2048
  bash benchmarks/e2e/run_profile_prefill_nvtx.sh /data/Qwen2.5-7B NVFP4 --batch-size 1
  bash benchmarks/e2e/run_profile_prefill_nvtx.sh --model /data/Llama-3.1-8B --no-nsys
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      POSITIONAL_MODE=1
      shift 2
      ;;
    --seq-len|--prefill-seq-len|--prefill_seq_len)
      PREFILL_SEQ_LEN="$2"
      shift 2
      ;;
    --batch-size|--batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --warmup-steps|--warmup_steps)
      WARMUP_STEPS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    -o|--out)
      OUT="${2%.nsys-rep}"
      shift 2
      ;;
    --no-conda)
      CONDA_ENV=""
      shift
      ;;
    --no-nsys)
      USE_NSYS=0
      shift
      ;;
    --trust-remote-code|--trust_remote_code)
      TRUST_REMOTE_CODE=1
      shift
      ;;
    --kv-cache|--kv_cache)
      KV_CACHE=1
      shift
      ;;
    --no-extra-fusion|--no_extra_fusion)
      NO_EXTRA_FUSION=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      PY_ARGS+=("$@")
      break
      ;;
    *)
      if [[ -z "${MODEL}" ]]; then
        MODEL="$1"
      elif [[ "${POSITIONAL_MODE}" -eq 0 ]]; then
        MODE="$1"
        POSITIONAL_MODE=1
      else
        PY_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  usage >&2
  exit 2
fi

case "${MODE}" in
  SHARQ|NVFP4|FP16) ;;
  sharq|nvfp4|fp16)
    MODE="$(echo "${MODE}" | tr '[:lower:]' '[:upper:]')"
    ;;
  *)
    echo "Unsupported --mode ${MODE}. Expected SHARQ, NVFP4, or FP16." >&2
    exit 2
    ;;
esac

MODEL_NAME="$(basename "${MODEL%/}")"
MODE_LOWER="$(echo "${MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ -z "${OUT}" ]]; then
  OUT="${REPO_ROOT}/profiles/${MODEL_NAME}_${MODE_LOWER}_prefill_s${PREFILL_SEQ_LEN}_b${BATCH_SIZE}"
fi
mkdir -p "$(dirname "${OUT}")"

PROFILE_CMD=(
  python "${SCRIPT_DIR}/profile_prefill_nvtx.py"
  --model "${MODEL}"
  --mode "${MODE}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --prefill_seq_len "${PREFILL_SEQ_LEN}"
  --warmup_steps "${WARMUP_STEPS}"
)

if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then
  PROFILE_CMD+=(--trust_remote_code)
fi
if [[ "${KV_CACHE}" -eq 1 ]]; then
  PROFILE_CMD+=(--kv_cache)
fi
if [[ "${NO_EXTRA_FUSION}" -eq 1 ]]; then
  PROFILE_CMD+=(--no_extra_fusion)
fi
PROFILE_CMD+=("${PY_ARGS[@]}")

cd "${REPO_ROOT}"

if [[ "${USE_NSYS}" -eq 1 ]]; then
  if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys was not found in PATH. Install Nsight Systems or re-run with --no-nsys." >&2
    exit 1
  fi

  CMD=(
    nsys profile
    --trace=cuda,nvtx,osrt,cublas,cudnn
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --force-overwrite=true
    -o "${OUT}"
    "${PROFILE_CMD[@]}"
    --capture_range
  )
else
  CMD=("${PROFILE_CMD[@]}")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

if [[ "${USE_NSYS}" -eq 1 ]]; then
  echo "Nsight report: ${OUT}.nsys-rep"
fi
