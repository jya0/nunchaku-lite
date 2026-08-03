"""FLUX.2 Klein 9B INT4/FP4 image generation and editing with Nunchaku Lite.

FLUX.2 Klein is a fast (4-step) text-to-image and image editing model built on
``Flux2Transformer2DModel``.  The KV variant (``Flux2KleinKVPipeline``) caches
reference-image K/V tensors for even faster multi-step editing.

Checkpoints:
    https://huggingface.co/tonera/FLUX.2-klein-9B-Nunchaku
    https://huggingface.co/tonera/FLUX.2-klein-9b-kv-Nunchaku

Run from the repository root:

    # Text-to-image (standard pipeline)
    python flux2_klein.py --prompt "a cat holding a sign that says hello world"

    # Image editing (standard pipeline)
    python flux2_klein.py --prompt "add sunglasses" --image photo.png

    # KV-cache pipeline (faster multi-step editing with reference image)
    python flux2_klein.py --prompt "add sunglasses" --image photo.png --kv

    # Interactive mode
    python flux2_klein.py --interactive

    # Batch generation
    python flux2_klein.py --prompt "cyberpunk city" --seed 42 --num 4

    # FP4 precision (Blackwell / RTX 50-series only)
    python flux2_klein.py --prompt "sunset over mountains" --precision fp4
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from nunchaku_lite import load_nunchaku_pipeline
from nunchaku_lite.utils import get_precision


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

DEFAULT_PRECISION = get_precision(device="cuda")
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
OUTPUT_DIR = Path("outputs")

# Model repos and checkpoints (INT4 for Ada / FP4 for Blackwell)
MODEL_IDS = {
    "standard": "tonera/FLUX.2-klein-9B-Nunchaku",
    "kv": "tonera/FLUX.2-klein-9b-kv-Nunchaku",
}
CHECKPOINT_NAMES = {
    "standard": {
        "int4": "svdq-int4_r32-FLUX.2-klein-9B-Nunchaku.safetensors",
        "fp4": "svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors",
    },
    "kv": {
        "int4": "svdq-int4_r32-FLUX.2-klein-9b-kv-Nunchaku.safetensors",
        "fp4": "svdq-fp4_r32-FLUX.2-klein-9b-kv-Nunchaku.safetensors",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FLUX.2 Klein 9B INT4/FP4 generation & editing with Nunchaku Lite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python flux2_klein.py --prompt 'a cat holding a sign that says hello world'\n"
        "  python flux2_klein.py --prompt 'add sunglasses' --image photo.png --kv\n"
        "  python flux2_klein.py --interactive\n"
        "  python flux2_klein.py --prompt 'cyberpunk city' --seed 42 --num 4\n",
    )
    parser.add_argument("--prompt", help="Text prompt for generation/editing.")
    parser.add_argument("--image", nargs="+", help="Input image path(s) for editing.")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt mode.")
    parser.add_argument("--kv", action="store_true", help="Use Flux2KleinKVPipeline (KV-cache, faster editing).")
    parser.add_argument("--precision", choices=["auto", "int4", "fp4"], default="auto", help="Quantization precision.")
    parser.add_argument("--rank", type=int, default=32, help="SVD quantization rank.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE, help="CFG guidance scale.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed (-1 for random).")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Output height.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Output width.")
    parser.add_argument("--num", type=int, default=1, help="Number of images to generate (text-to-image only).")
    parser.add_argument("--model-id", default=None, help="Override HF model repo ID.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path/repo.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
    parser.add_argument(
        "--quantized-encoder",
        action="store_true",
        help="Use 4-bit quantized Qwen3 text encoder (nunchaku-qwen3 awq-int4-flux.2-qwen3).",
    )
    parser.add_argument(
        "--force-cfg",
        action="store_true",
        help="Override is_distilled=false on standard pipeline to enable classifier-free guidance (CFG). "
        "Required for anatomy correction edits — without CFG the edit prompt has minimal effect on distilled models. "
        "Only works with standard pipeline (not --kv).",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt for CFG (only used with --force-cfg on standard pipeline). "
        "Describes what to avoid, e.g. 'extra limbs, extra fingers, deformed hands, fused fingers, feet as hands'.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Custom tag appended to output filename (e.g. 'anatomy', 'cfg2'). Defaults to None.",
    )
    return parser.parse_args()


def resolve_precision(precision_arg: str) -> str:
    if precision_arg != "auto":
        return precision_arg
    return DEFAULT_PRECISION


def resolve_model_and_checkpoint(args: argparse.Namespace, precision: str) -> tuple[str, str]:
    variant = "kv" if args.kv else "standard"
    model_id = args.model_id or MODEL_IDS[variant]
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        ckpt_name = CHECKPOINT_NAMES[variant][precision]
        checkpoint = f"{model_id}/{ckpt_name}"
    return model_id, checkpoint


def resolve_pipeline_cls(args: argparse.Namespace):
    from diffusers import Flux2KleinKVPipeline, Flux2KleinPipeline

    if args.kv:
        return Flux2KleinKVPipeline
    return Flux2KleinPipeline


def load_pipeline(args: argparse.Namespace):
    precision = resolve_precision(args.precision)
    model_id, checkpoint = resolve_model_and_checkpoint(args, precision)
    pipeline_cls = resolve_pipeline_cls(args)

    quantized_encoder_checkpoints = None
    if args.quantized_encoder:
        quantized_encoder_checkpoints = {
            "text_encoder": "mit-han-lab/nunchaku-qwen3/awq-int4-flux.2-qwen3.safetensors",
        }

    pipe = load_nunchaku_pipeline(
        model_id,
        pipeline_cls=pipeline_cls,
        checkpoint=checkpoint,
        target="flux2",
        precision=precision,
        torch_dtype=torch.bfloat16,
        device="cuda",
        quantized_encoder_checkpoints=quantized_encoder_checkpoints,
    )

    # The Klein model ships with is_distilled=True, which disables CFG entirely
    # (do_classifier_free_guidance returns False).  For anatomy-correction edits,
    # CFG is essential — without it the edit prompt has almost no effect and the
    # model just reconstructs the input image.  --force-cfg overrides this.
    if args.force_cfg and not args.kv:
        pipe.config.is_distilled = False
        print("  [force-cfg] Override is_distilled=false — CFG enabled (guidance_scale will take effect)", flush=True)

    pipe = pipe.to("cuda")
    return pipe


def load_images(args: argparse.Namespace) -> list[Path] | None:
    if not args.image:
        return None
    images = [Path(p).expanduser() for p in args.image]
    for img in images:
        if not img.is_file():
            raise FileNotFoundError(f"image not found: {img}")
    return images


def generate_text_to_image(
    pipe,
    prompt: str,
    args: argparse.Namespace,
    output_dir: Path,
    idx: int = 0,
) -> Path:
    seed = args.seed + idx if args.seed >= 0 else int(torch.Generator(device="cuda").random_().item())
    generator = torch.Generator(device="cuda").manual_seed(seed)

    t0 = time.time()
    result = pipe(
        prompt=prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    gen_time = time.time() - t0
    output_image = result.images[0]

    output_path = output_dir / f"flux2klein_t2i_s{seed}_g{args.guidance_scale}_{idx}.png"
    output_image.save(output_path)
    print(
        f"  [{idx+1}] -> {output_path.name}  "
        f"({args.height}x{args.width}, {args.steps} steps, g={args.guidance_scale}, {gen_time:.1f}s)",
        flush=True,
    )
    return output_path


def edit_image(
    pipe,
    image_path: Path,
    prompt: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> Path:
    input_image = Image.open(str(image_path)).convert("RGB")
    seed = args.seed if args.seed >= 0 else int(torch.Generator(device="cuda").random_().item())
    generator = torch.Generator(device="cuda").manual_seed(seed)

    call_kwargs = {
        "image": input_image,
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "generator": generator,
    }
    # KV pipeline has no guidance_scale parameter (CFG is not supported).
    # Standard pipeline accepts guidance_scale, but it's a no-op unless
    # is_distilled is overridden to False (see --force-cfg in load_pipeline).
    suffix = "kv" if args.kv else "std"
    if not args.kv:
        call_kwargs["guidance_scale"] = args.guidance_scale
        # Encode negative prompt manually for CFG.  The pipeline hardcodes
        # negative_prompt="" internally, but we can pre-encode and pass
        # negative_prompt_embeds to override it with a real negative prompt.
        if args.force_cfg and args.negative_prompt:
            neg_embeds, neg_text_ids = pipe.encode_prompt(
                prompt=args.negative_prompt,
                device=pipe._execution_device,
                num_images_per_prompt=1,
            )
            call_kwargs["negative_prompt_embeds"] = neg_embeds
            # The pipeline also needs negative_text_ids — store on pipe for the
            # __call__ path.  It reads negative_prompt_embeds from kwargs and
            # re-encodes if None, but if we pass embeds it uses them directly.
            # However, text_ids are derived internally from the embeds shape,
            # so passing embeds alone is sufficient.
            suffix = f"std_g{args.guidance_scale}"

    if args.tag:
        suffix = f"{suffix}_{args.tag}"

    t0 = time.time()
    result = pipe(**call_kwargs)
    gen_time = time.time() - t0
    output_image = result.images[0]

    output_path = output_dir / f"{image_path.stem}_klein_s{seed}_{suffix}.png"
    output_image.save(output_path)
    neg_info = f", neg='{args.negative_prompt[:40]}...'" if (args.negative_prompt and not args.kv) else ""
    print(
        f"  {image_path.name} -> {output_path.name}  "
        f"({args.height}x{args.width}, {args.steps} steps, {'kv' if args.kv else f'g={args.guidance_scale}'}, "
        f"{gen_time:.1f}s{neg_info})",
        flush=True,
    )
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant = "KV-cache" if args.kv else "standard"
    print(f"Loading FLUX.2 Klein 9B pipeline ({variant})...", flush=True)
    pipe = load_pipeline(args)

    if args.interactive:
        while True:
            prompt = input("\nprompt (empty to quit)> ").strip()
            if not prompt:
                break
            image_paths = load_images(args)
            if image_paths:
                for img_path in image_paths:
                    edit_image(pipe, img_path, prompt, args, output_dir)
            else:
                generate_text_to_image(pipe, prompt, args, output_dir)
    else:
        if not args.prompt:
            raise ValueError("--prompt is required in non-interactive mode.")
        image_paths = load_images(args)
        if image_paths:
            for img_path in image_paths:
                edit_image(pipe, img_path, args.prompt, args, output_dir)
        else:
            for i in range(args.num):
                generate_text_to_image(pipe, args.prompt, args, output_dir, idx=i)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
