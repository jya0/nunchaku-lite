#!/usr/bin/env bash
#
# SUPIR batch image upscaler / restorer
#
# Usage:
#   ./supir_upscale.sh <input_folder> <output_folder> [options]
#
# Options:
#   --upscale N          Upscale factor (default: 2)
#   --sign Q|F           Model variant: Q=quality, F=fidelity (default: Q)
#   --steps N            EDM sampling steps (default: 50)
#   --cfg F              CFG scale (default: 4.0; quality=6.0, fidelity=4.0)
#   --no-llava           Skip LLaVA captioning (faster, less VRAM, no auto-prompt)
#   --8bit-llava         Load LLaVA in 8-bit (saves VRAM, ~12GB)
#   --tile-vae           Use tiled VAE (saves VRAM for large outputs)
#   --half               Load SUPIR in half precision (fp16)
#   --min-size N         Minimum output resolution (default: 1024)
#   --color-fix TYPE     Color fix: None, AdaIn, Wavelet (default: Wavelet)
#   --seed N             Random seed (default: 1234)
#   --s-stage2 F         Stage2 control strength (default: 1.0; 0.93=more quality)
#   --device GPU         CUDA device id (default: 0)
#
# Examples:
#   # Basic 2x upscale with quality model
#   ./supir_upscale.sh ./input_images ./output_images
#
#   # Fidelity model, skip LLaVA (lower VRAM), 4x upscale
#   ./supir_upscale.sh ./inputs ./outputs --sign F --no-llava --upscale 4
#
#   # Quality-oriented, 8-bit LLaVA, tiled VAE (RTX 4090 friendly)
#   ./supir_upscale.sh ./inputs ./outputs --sign Q --cfg 6.0 --8bit-llava --tile-vae

set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPIR_DIR="${SCRIPT_DIR}/thirdparty/SUPIR"
VENV_PY="${SCRIPT_DIR}/.venv-supir/bin/python"

if [ ! -f "$VENV_PY" ]; then
    echo "ERROR: SUPIR venv not found at $VENV_PY" >&2
    echo "Create it with: uv venv --python 3.8 .venv-supir && uv pip install ..." >&2
    exit 1
fi

if [ ! -d "$SUPIR_DIR" ]; then
    echo "ERROR: SUPIR submodule not found at $SUPIR_DIR" >&2
    echo "Initialize with: git submodule update --init --recursive" >&2
    exit 1
fi

# ─── Parse args ──────────────────────────────────────────────────────────────
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder> [options]" >&2
    exit 1
fi

INPUT_DIR="$1"; shift
OUTPUT_DIR="$1"; shift

# Defaults
UPSCALE=2
SIGN="Q"
STEPS=50
CFG=4.0
NO_LLAVA=false
LLAVA_8BIT=false
TILE_VAE=false
HALF=false
MIN_SIZE=1024
COLOR_FIX="Wavelet"
SEED=1234
S_STAGE2=1.0
DEVICE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --upscale)    UPSCALE="$2"; shift 2 ;;
        --sign)       SIGN="$2"; shift 2 ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --cfg)        CFG="$2"; shift 2 ;;
        --no-llava)   NO_LLAVA=true; shift ;;
        --8bit-llava) LLAVA_8BIT=true; shift ;;
        --tile-vae)   TILE_VAE=true; shift ;;
        --half)       HALF=true; shift ;;
        --min-size)   MIN_SIZE="$2"; shift 2 ;;
        --color-fix)  COLOR_FIX="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --s-stage2)   S_STAGE2="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        -h|--help)    head -30 "$0"; exit 0 ;;
        *)            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ─── Validate ────────────────────────────────────────────────────────────────
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input folder does not exist: $INPUT_DIR" >&2
    exit 1
fi

# Count input images
IMG_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.bmp' \) | wc -l)
if [ "$IMG_COUNT" -eq 0 ]; then
    echo "ERROR: No images (png/jpg/jpeg/webp/bmp) found in $INPUT_DIR" >&2
    exit 1
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  SUPIR Batch Upscaler                                    ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Input:      %-44s║\n" "$INPUT_DIR"
printf "║  Output:     %-44s║\n" "$OUTPUT_DIR"
printf "║  Images:     %-44d║\n" "$IMG_COUNT"
printf "║  Model:      %-44s║\n" "SUPIR-v0${SIGN} (${SIGN}=quality/fidelity)"
printf "║  Upscale:    %-44dx║\n" "$UPSCALE"
printf "║  Steps:      %-44d║\n" "$STEPS"
printf "║  CFG:        %-44s║\n" "$CFG"
printf "║  Min size:   %-44d║\n" "$MIN_SIZE"
printf "║  Color fix:  %-44s║\n" "$COLOR_FIX"
printf "║  Seed:       %-44d║\n" "$SEED"
printf "║  s_stage2:   %-44s║\n" "$S_STAGE2"
printf "║  LLaVA:      %-44s║\n" "$( [ "$NO_LLAVA" = true ] && echo 'disabled' || ( [ "$LLAVA_8BIT" = true ] && echo '8-bit' || echo 'full (16-bit)' ) )"
printf "║  Tile VAE:   %-44s║\n" "$( [ "$TILE_VAE" = true ] && echo 'yes' || echo 'no' )"
printf "║  Half:       %-44s║\n" "$( [ "$HALF" = true ] && echo 'yes' || echo 'no' )"
printf "║  GPU:        %-44s║\n" "cuda:$DEVICE"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── Build test.py args ──────────────────────────────────────────────────────
ARGS=(
    --img_dir "$INPUT_DIR"
    --save_dir "$OUTPUT_DIR"
    --upscale "$UPSCALE"
    --SUPIR_sign "$SIGN"
    --edm_steps "$STEPS"
    --s_cfg "$CFG"
    --min_size "$MIN_SIZE"
    --color_fix_type "$COLOR_FIX"
    --seed "$SEED"
    --s_stage2 "$S_STAGE2"
)

if [ "$NO_LLAVA" = true ]; then
    ARGS+=(--no_llava)
elif [ "$LLAVA_8BIT" = true ]; then
    ARGS+=(--load_8bit_llava)
fi

if [ "$TILE_VAE" = true ]; then
    ARGS+=(--use_tile_vae)
fi

if [ "$HALF" = true ]; then
    ARGS+=(--loading_half_params)
fi

# ─── Run ─────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
cd "$SUPIR_DIR"
export CUDA_VISIBLE_DEVICES="$DEVICE"
echo "Running: CUDA_VISIBLE_DEVICES=$DEVICE python test.py ${ARGS[*]}"
echo ""
exec "$VENV_PY" test.py "${ARGS[@]}"
