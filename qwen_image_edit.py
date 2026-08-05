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

# Model registry: each entry maps model key -> HF pipeline id and {rank: checkpoint path}.
#   qwen-2509: official nunchaku-tech lightning 8-step distilled checkpoints.
#   qwen-2511: QuantFunc re-quantization of Qwen-Image-Edit-2511 (same QwenImageTransformer
#              architecture, so the same nunchaku-lite "qwen_image" adapter loads it unchanged).
MODELS = {
    "2509": {
        "model_id": "Qwen/Qwen-Image-Edit-2509",
        "ckpt": {
            "r128": Path(__file__).resolve().parent / "ckpts" / "qwen-image-edit-2509" / "lightning-251115"
                    / "svdq-int4_r128-qwen-image-edit-2509-lightning-8steps-251115.safetensors",
            "r32": Path(__file__).resolve().parent / "ckpts" / "qwen-image-edit-2509" / "lightning-251115"
                   / "svdq-int4_r32-qwen-image-edit-2509-lightning-8steps-251115.safetensors",
        },
    },
    "2511": {
        "model_id": "Qwen/Qwen-Image-Edit-2511",
        "ckpt": {
            # QuantFunc variants (rank in the low-rank quantization):
            #   best_quality   = rank 256 (highest quality)
            #   balance        = rank 128
            #   ultimate_speed = rank 32
            "r256": Path(__file__).resolve().parent / "ckpts" / "qwen-image-edit-2511"
                    / "nunchaku_qwen_image_edit_2511_best_quality_int4.safetensors",
            "r128": Path(__file__).resolve().parent / "ckpts" / "qwen-image-edit-2511"
                    / "nunchaku_qwen_image_edit_2511_balance_int4.safetensors",
            "r32": Path(__file__).resolve().parent / "ckpts" / "qwen-image-edit-2511"
                   / "nunchaku_qwen_image_edit_2511_ultimate_speed_int4.safetensors",
        },
    },
}

# QuantFunc 2511 checkpoints are NOT lightning-distilled (full 40-step denoising model),
# so a different default step count / CFG applies. 2509 keeps the lightning 8-step defaults.
DEFAULT_STEPS = {"2509": 8, "2511": 40}
DEFAULT_CFG = {"2509": 2.0, "2511": 4.0}

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
    "foot": (
        "Fix the foot. Render a natural, correctly proportioned foot with EXACTLY five toes, "
        "evenly spaced and properly aligned. Remove any sixth toe or extra digit, and remove "
        "any fused or misshapen toes. Keep everything else exactly as it is."
    ),
    "body": (
        "Fix the distorted limbs and body anatomy. Correct hands, feet, and limbs to natural "
        "proportions, restoring any missing feet or hands. Keep the face and expression unchanged."
    ),
}

# Negative prompt: artifact-focused, 3-6 terms (deAPI best practice).
# Lightning models DO process negative prompts — keep them short and targeted.
DEFAULT_NEGATIVE_PROMPT = "extra fingers, missing fingers, fused fingers, deformed limbs, extra toes, deformed feet, bad anatomy"


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
        description="Qwen-Image-Edit INT4 anatomy correction with Nunchaku Lite (2509 or 2511)."
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
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()), default="2511",
                        help="Which Qwen-Image-Edit model to use (2509 or 2511)")
    parser.add_argument("--rank", type=str, choices=["32", "128", "256"], default="256",
                        help="Quantization rank (256=best quality, 128=balanced, 32=speed)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of inference steps (default: 8 for 2509 lightning, 40 for 2511 full)")
    parser.add_argument("--fast", action="store_true",
                        help="Lightning-style fast mode for 2511: uses the AuraFlow time-shift scheduler "
                             "(shift=3) like QuantFunc's official 4-step workflow. Defaults steps=4, CFG=1.0. "
                             "Much faster than the default 40-step full run.")
    parser.add_argument("--true-cfg-scale", type=float, default=None,
                        help="CFG scale (1.0=passthrough, 2.0=balanced edit, 4.0=strong but may burn)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to an NSFW/adapter LoRA to load at runtime (e.g. SexGod 2511)")
    parser.add_argument("--lora-weight", type=float, default=1.0,
                        help="LoRA adapter weight (Phr00t suggests ~0.6-1.0)")

    # Output size (defaults to input image size)
    parser.add_argument("--width", type=int, default=None, help="Output width (default: input image width)")
    parser.add_argument("--height", type=int, default=None, help="Output height (default: input image height)")

    # Crop-and-recompose: edit ONLY a crop region and feather-blend it back into the
    # ORIGINAL image. This is the workaround for the fact that Qwen-Image-Edit has no
    # masking/inpainting support. It lets a targeted artifact (e.g. a deformed foot) get
    # full denoising attention, while leaving everything outside the crop pixel-identical
    # to the source (so a LoRA can't re-draw a vagina that isn't in the crop).
    parser.add_argument("--crop", type=str, default=None, metavar="X,Y,W,H",
                        help="Crop box 'x,y,w,h' in input-image coords. If set, ONLY this region is "
                             "edited (upscaled to 1024 if smaller), then feathered back onto the original. "
                             "Leave unset for whole-image editing.")
    parser.add_argument("--crop-blend", type=float, default=0.15,
                        help="Feather fraction (of min(crop w,h)) for blending the corrected crop back (default 0.15)")

    return parser.parse_args()


def load_pipeline(model: str, rank: str, lora: str | None = None, lora_weight: float = 1.0,
                  fast: bool = False) -> QwenImageEditPlusPipeline:
    """Load the Qwen-Image-Edit pipeline with Nunchaku Lite INT4 weights.

    Args:
        model: Model key in MODELS ("2509" or "2511").
        rank: Quantization rank ("32", "128", "256").
        lora: Optional path to an NSFW/adapter LoRA to load at runtime.
        lora_weight: Strength for the LoRA (e.g. Phr00t uses ~0.6-1.0).
        fast: Use the lightning/AuraFlow time-shift scheduler (needed for 2511 low-step runs).
    """
    model_cfg = MODELS[model]
    model_id = model_cfg["model_id"]
    checkpoint = model_cfg["ckpt"][f"r{rank}"]

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"For 2509, download from nunchaku-tech/nunchaku-qwen-image-edit-2509.\n"
            f"For 2511, download from QuantFunc/Nunchaku-Qwen-Image-EDIT-2511."
        )

    print(f"Loading Qwen-Image-Edit-{model} INT4 {rank} ({checkpoint.name})...", flush=True)
    print(f"  Checkpoint: {checkpoint}", flush=True)

    # 2509 uses the lightning-distilled scheduler config; 2511 is the full (non-distilled) model.
    # For 2511 in --fast mode, use the same AuraFlow time-shift (shift=3) scheduler that QuantFunc's
    # official 4-step workflow uses, which enables low-step (4-step) inference on the full model.
    if model == "2509" or (model == "2511" and fast):
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(lightning_scheduler_config())
    else:
        scheduler = None

    t0 = time.time()
    if scheduler is None:
        pipe = load_nunchaku_pipeline(
            model_id,
            pipeline_cls=QwenImageEditPlusPipeline,
            checkpoint=str(checkpoint),
            target="qwen_image",
            precision="int4",
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = load_nunchaku_pipeline(
            model_id,
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

    if lora:
        lora_path = Path(lora)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_path}")
        print(f"  Loading LoRA: {lora_path} (weight={lora_weight})", flush=True)
        t1 = time.time()
        pipe.load_lora_weights(str(lora_path), adapter_name="nsfw")
        pipe.set_adapters("nsfw", adapter_weights=lora_weight)
        print(f"  LoRA loaded in {time.time()-t1:.1f}s", flush=True)

    return pipe


def _parse_crop(crop: str, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Parse 'x,y,w,h' crop box and clamp to image bounds."""
    try:
        x, y, w, h = (int(v) for v in crop.split(","))
    except ValueError:
        raise ValueError(f"--crop must be 'x,y,w,h', got {crop!r}")
    if w <= 0 or h <= 0:
        raise ValueError(f"--crop w/h must be > 0, got {crop!r}")
    # Clamp to image bounds (x,y are top-left).
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return x, y, w, h


def _feathered_paste(
    base: "Image.Image", crop_edited: "Image.Image", x: int, y: int, w: int, h: int, blend: float
) -> "Image.Image":
    """Blend `crop_edited` back onto `base` at (x,y) with a feathered (soft) edge.

    Uses a distance-from-edge alpha mask so the corrected crop merges seamlessly into
    the original — no visible seam at the crop boundary. `blend` is the feather width
    as a fraction of min(w, h).
    """
    from PIL import Image, ImageFilter
    import numpy as np

    # Distance-from-edge alpha: 0 at the border, 1 in the interior.
    feather = max(1, int(blend * min(w, h)))
    edge = np.ones((h, w), dtype=np.float32)
    edge[feather:-feather, feather:-feather] = 0.0
    edge = Image.fromarray((edge * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(feather)
    )
    alpha = np.asarray(edge, dtype=np.float32)[..., None] / 255.0

    base_arr = np.asarray(base).astype(np.float32)
    edited_arr = np.asarray(crop_edited).astype(np.float32)
    region = base_arr[y:y + h, x:x + w]
    blended = region * (1.0 - alpha) + edited_arr * alpha
    base.paste(Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)), (x, y))
    return base


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
    crop: str | None = None,
    crop_blend: float = 0.15,
) -> Path:
    """Run a single image edit and save the result.

    If `crop` ('x,y,w,h') is given, only that region is edited (upscaled to a minimum
    1024 on the long side for quality), then feathered back onto the ORIGINAL image.
    Everything outside the crop stays pixel-identical to the input.
    """
    image = load_image(image_path).convert("RGB")

    # Default to input image dimensions
    out_width = width or image.width
    out_height = height or image.height

    generator = torch.Generator(device="cuda").manual_seed(seed)

    edit_target = image
    paste_box = None

    if crop:
        x, y, w, h = _parse_crop(crop, image.width, image.height)
        print(f"  CROP mode: editing only region ({x},{y},{w},{h}), blending back with feather={crop_blend}",
              flush=True)
        # Upscale the crop to a reasonable edit resolution (keep aspect, min side 1024).
        scale = 1024 / min(w, h)
        cw, ch = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
        edit_target = image.crop((x, y, x + w, y + h)).resize((cw, ch), Image.LANCZOS)
        paste_box = (x, y, w, h)
        out_width, out_height = cw, ch

    print(f"  Input: {image_path} ({image.width}x{image.height})", flush=True)
    print(f"  Output: {out_width}x{out_height}, {steps} steps, cfg={true_cfg_scale}", flush=True)
    print(f"  Prompt: {prompt}", flush=True)

    t0 = time.time()
    result = pipe(
        image=edit_target,
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

    if crop:
        # Resize the edited crop back to the original crop size and feather-blend in.
        crop_edited = output_image.resize((w, h), Image.LANCZOS)
        image = _feather_paste(image, crop_edited, x, y, w, h, crop_blend)
        output_image = image

    # Generate output filename (include CFG to prevent overwrites in sweeps).
    # A short hash of the prompt disambiguates different prompts sharing seed/cfg.
    import hashlib
    prompt_tag = hashlib.md5(prompt.encode()).hexdigest()[:6]
    input_stem = Path(image_path).stem
    suffix = f"_crop{x}_{y}_{w}_{h}" if crop else ""
    output_path = output_dir / f"{input_stem}_edited_s{seed}_cfg{true_cfg_scale}{suffix}_{prompt_tag}.png"
    output_image.save(output_path)

    print(f"  Generated in {gen_time:.1f}s -> {output_path}", flush=True)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve per-model defaults for steps/CFG (2509 lightning uses few steps; 2511 is full 40-step).
    # In --fast mode, match QuantFunc's official 4-step workflow (steps=4, CFG=1.0).
    if args.fast and args.model == "2511":
        if args.steps is None:
            args.steps = 4
        if args.true_cfg_scale is None:
            args.true_cfg_scale = 1.0
    else:
        if args.steps is None:
            args.steps = DEFAULT_STEPS[args.model]
        if args.true_cfg_scale is None:
            args.true_cfg_scale = DEFAULT_CFG[args.model]

    pipe = load_pipeline(args.model, args.rank, args.lora, args.lora_weight, args.fast)

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
                args.crop, args.crop_blend,
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
                args.crop, args.crop_blend,
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
                args.crop, args.crop_blend,
            )


if __name__ == "__main__":
    main()
