#!/usr/bin/env python3
"""Qwen-Image-Edit-2509 INT4 with Nunchaku Lite — anatomy correction pipeline.

Uses the lightning-251115 8-step distilled weights (V1.0 LoRA fused, edit-specific)
for fast 8-step inference on RTX 4090 (Ada / sm_89).

Checkpoints (local):
    ckpts/qwen-image-edit-2509/lightning-251115/
        svdq-int4_r128-qwen-image-edit-2509-lightning-8steps-251115.safetensors  (default, better quality)
        svdq-int4_r32-qwen-image-edit-2509-lightning-8steps-251115.safetensors   (faster, smaller)

Usage:
    # Default: r128, 8 steps, single image
    python qwen_image_edit.py --image input.png --prompt "fix anatomy, correct body proportions"

    # Multiple images (model loads once, processes all)
    python qwen_image_edit.py --image img1.png img2.png img3.png --prompt "fix anatomy"

    # Use r32 instead
    python qwen_image_edit.py --image input.png --prompt "fix anatomy" --rank 32

    # Batch: process all images in a directory
    python qwen_image_edit.py --input-dir images/ --prompt "fix anatomy" --output-dir corrected/

    # Interactive mode (prompt for each image)
    python qwen_image_edit.py --image input.png --interactive

Pipeline context:
    Input (1024x1024) → Qwen-Image-Edit (fix anatomy) → corrected image → SeedVR2 (upscale to 4096x4096)
"""

import argparse
import math
import time
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku_lite import load_nunchaku_pipeline


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
CKPT_DIR = Path(__file__).resolve().parent / "ckpts" / "qwen-image-edit-2509" / "lightning-251115"

CKPT_BY_RANK = {
    128: CKPT_DIR / "svdq-int4_r128-qwen-image-edit-2509-lightning-8steps-251115.safetensors",
    32: CKPT_DIR / "svdq-int4_r32-qwen-image-edit-2509-lightning-8steps-251115.safetensors",
}

NUM_INFERENCE_STEPS = 8  # lightning 8-step distilled model
TRUE_CFG_SCALE = 2.0    # 1.0=no CFG (passthrough), 2.0=balanced edit (works), 4.0=strong but burns

# 4-Part Edit Prompt Formula (Operation + Target + Specification + Preservation):
# Qwen-Image-Edit expects edit instructions, NOT generator descriptions.
# One operation per prompt. Preservation clause must be scoped (not all-encompassing).
# Lightning distilled: keep preservation clauses SHORT — over-locking suppresses the edit.
PROMPT_PRESETS = {
    "hands": (
        "Fix the hands. Render correct hands with exactly five fingers, natural joints, and "
        "proper proportions, removing any extra digits or fused fingers. Keep the face and "
        "expression unchanged."
    ),
    "body": (
        "Fix the distorted limbs and body anatomy. Correct hands, feet, and limbs to natural "
        "proportions, restoring any missing feet or hands. Keep the face and expression unchanged."
    ),
}

# Negative prompt: artifact-focused, 3-6 terms (deAPI best practice).
# Lightning models DO process negative prompts — keep them short and targeted.
DEFAULT_NEGATIVE_PROMPT = "extra fingers, missing fingers, fused fingers, deformed limbs, bad anatomy"


def lightning_scheduler_config() -> dict:
    """Scheduler config for Qwen-Image-Lightning distilled models.

    From: https://github.com/ModelTC/Qwen-Image-Lightning
    Key: shift=3 (log(3)) used during distillation, dynamic shifting enabled.
    """
    return {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Edit-2509 INT4 anatomy correction with Nunchaku Lite."
    )
    # Input options (mutually exclusive: one or more images, directory, or interactive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, nargs="+", help="One or more input image paths")
    input_group.add_argument("--input-dir", type=str, help="Directory of images to process")
    input_group.add_argument("--interactive", action="store_true", help="Interactive mode: load image, type prompt")

    parser.add_argument("--output-dir", type=str, default="outputs/qwen_edit", help="Output directory")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Edit prompt. If omitted, --preset is used.")
    parser.add_argument("--preset", type=str, choices=list(PROMPT_PRESETS.keys()), default="hands",
                        help="Prompt preset to use when --prompt is not given (default: hands)")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt (artifact-focused; defaults to hand/limb defects)")

    # Model options
    parser.add_argument("--rank", type=int, choices=[32, 128], default=128, help="Quantization rank (128=quality, 32=speed)")
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS, help="Number of inference steps")
    parser.add_argument("--true-cfg-scale", type=float, default=TRUE_CFG_SCALE,
                        help="CFG scale (1.0=passthrough, 2.0=balanced edit, 4.0=strong but may burn)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output size (defaults to input image size)
    parser.add_argument("--width", type=int, default=None, help="Output width (default: input image width)")
    parser.add_argument("--height", type=int, default=None, help="Output height (default: input image height)")

    return parser.parse_args()


def load_pipeline(rank: int) -> QwenImageEditPlusPipeline:
    """Load the Qwen-Image-Edit pipeline with Nunchaku Lite INT4 weights."""
    checkpoint = CKPT_BY_RANK[rank]
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Download with: python -c \"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download('nunchaku-tech/nunchaku-qwen-image-edit-2509', "
            f"'lightning-251115/{checkpoint.name}', local_dir='ckpts/qwen-image-edit-2509')\""
        )

    print(f"Loading Qwen-Image-Edit-2509 INT4 r{rank} (lightning 8-step 251115)...", flush=True)
    print(f"  Checkpoint: {checkpoint}", flush=True)

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(lightning_scheduler_config())

    t0 = time.time()
    pipe = load_nunchaku_pipeline(
        MODEL_ID,
        pipeline_cls=QwenImageEditPlusPipeline,
        checkpoint=str(checkpoint),
        target="qwen_image",
        precision="int4",
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")
    load_time = time.time() - t0
    print(f"  Pipeline loaded in {load_time:.1f}s", flush=True)

    return pipe


def edit_image(
    pipe: QwenImageEditPlusPipeline,
    image_path: str,
    prompt: str,
    negative_prompt: str,
    steps: int,
    true_cfg_scale: float,
    seed: int,
    width: int | None,
    height: int | None,
    output_dir: Path,
) -> Path:
    """Run a single image edit and save the result."""
    image = load_image(image_path).convert("RGB")

    # Default to input image dimensions
    out_width = width or image.width
    out_height = height or image.height

    generator = torch.Generator(device="cuda").manual_seed(seed)

    print(f"  Input: {image_path} ({image.width}x{image.height})", flush=True)
    print(f"  Output: {out_width}x{out_height}, {steps} steps, cfg={true_cfg_scale}", flush=True)
    print(f"  Prompt: {prompt}", flush=True)

    t0 = time.time()
    result = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=out_width,
        height=out_height,
        num_inference_steps=steps,
        true_cfg_scale=true_cfg_scale,
        generator=generator,
    )
    gen_time = time.time() - t0

    output_image = result.images[0]

    # Generate output filename (include CFG to prevent overwrites in sweeps)
    input_stem = Path(image_path).stem
    output_path = output_dir / f"{input_stem}_edited_s{seed}_cfg{true_cfg_scale}.png"
    output_image.save(output_path)

    print(f"  Generated in {gen_time:.1f}s -> {output_path}", flush=True)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args.rank)

    if args.interactive:
        # Interactive mode: load image path from arg, prompt from stdin
        if not args.image:
            print("--interactive requires --image")
            return
        # In interactive mode, only use the first image
        args.image = [args.image[0]]
        while True:
            try:
                prompt = input("\nedit prompt (or 'quit')> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye.")
                break
            if not prompt or prompt.lower() in {"quit", "exit", "q"}:
                print("bye.")
                break
            edit_image(
                pipe, args.image, prompt, args.negative_prompt,
                args.steps, args.true_cfg_scale, args.seed,
                args.width, args.height, output_dir,
            )
        return

    if not args.prompt:
        args.prompt = PROMPT_PRESETS[args.preset]
        print(f"Using preset '{args.preset}': {args.prompt}", flush=True)

    if args.image:
        images = [Path(p) for p in args.image]
        print(f"\nProcessing {len(images)} image(s)...", flush=True)
        for img_path in images:
            print(f"\n--- {img_path.name} ---", flush=True)
            edit_image(
                pipe, str(img_path), args.prompt, args.negative_prompt,
                args.steps, args.true_cfg_scale, args.seed,
                args.width, args.height, output_dir,
            )
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        images = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in extensions)
        if not images:
            print(f"No images found in {input_dir}")
            return
        print(f"\nProcessing {len(images)} images from {input_dir}...", flush=True)
        for img_path in images:
            print(f"\n--- {img_path.name} ---", flush=True)
            edit_image(
                pipe, str(img_path), args.prompt, args.negative_prompt,
                args.steps, args.true_cfg_scale, args.seed,
                args.width, args.height, output_dir,
            )


if __name__ == "__main__":
    main()
