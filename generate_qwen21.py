"""Qwen-Image-2.1 INT4 (nunchaku-lite) interactive generation loop.

Checkpoint (local assembled diffusers pipeline dir):
    ckpts/qwen-image-2.1/
        lite/diffusion_pytorch_model.safetensors   nunchaku_lite INT4 DiT (224 quantized linears)
        text_encoder_fp8/...FP8.safetensors         Qwen3-VL-8B text encoder, FP8
        vae/, scheduler/, processor/, model_index.json

Run from the repository root:

    python generate_qwen21.py

Type a prompt at each `prompt>` line and press Enter to generate an image.
Then enter a seed at `seed>`: `-1` (or empty) for random, or a specific integer
to reproduce a fixed result. Type `quit` (or Ctrl+C) to exit.
All images are saved under outputs/ with their seed in the filename.

Image editing (up to 10 reference images; order matters):
    python generate_qwen21.py --image a.png
    python generate_qwen21.py --image a.png b.png c.png

Notes on this model (verified 2026-10-27 on RTX 4090 / sm_89):
  * Guidance is OFF by default and that is intentional. `true_cfg_scale` only
    engages when a negative prompt is ALSO given, and it then runs the DiT twice
    per step (~2.2x slower). Official Qwen-Image-2.1 docs never use one.
  * `use_kv_cache=True` is a 2.6x speedup on multi-reference edits. Keep it on.
  * 1024x1024 peaks at ~27.5 GiB. The official default is 2048x2048, which does
    NOT fit comfortably in 48 GB (~48.2/49.1 GiB, thrashes) — 1024 is the
    practical local size.
  * LoRA is NOT supported here: diffusers' NunchakuLiteQuantizer swaps linears
    for `SVDQW4A4Linear` (a plain nn.Module), and PEFT only accepts nn.Linear.
  * Needs `DIFFUSERS_TRUST_REMOTE_KERNELS=true` because the quantizer fetches
    kernels from `rootonchair/nunchaku-lite-kernels`, which is not a trusted
    publisher. Set automatically below.
"""

import os

# Must be set before the quantizer runs (i.e. before from_pretrained), so do it
# before importing diffusers.
os.environ.setdefault("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")

from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from diffusers.utils import load_image


model_id = str(Path(__file__).resolve().parent / "ckpts" / "qwen-image-2.1")
precision = "int4"
output_dir = Path("outputs")

# Fixed generation settings (per request) — official Qwen-Image-2.1 defaults,
# except the resolution (see note above; official is 2048x2048).
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 40
TRUE_CFG_SCALE = 1.0  # 1.0 = guidance off (official). >1 needs a negative prompt.
USE_KV_CACHE = True  # 2.6x faster on multi-reference edits
MAX_REFERENCE_IMAGES = 10  # diffusers allows up to 10 (vLLM caps at 4)

# Official aspect ratios, scaled to ~1024^2 area. Every value is divisible by 32
# (the pipeline's `multiple_of`), so nothing gets silently rounded.
ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "4:3": (1184, 896),
    "3:4": (896, 1184),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "16:9": (1376, 768),
    "9:16": (768, 1376),
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Qwen-Image-2.1 INT4 interactive generation.")
    parser.add_argument("--image", type=str, nargs="+", default=None,
                        help=f"Reference image(s) for editing, in reading order (up to {MAX_REFERENCE_IMAGES})")
    parser.add_argument("--aspect", type=str, choices=list(ASPECT_RATIOS.keys()), default="1:1",
                        help="Output aspect ratio (default: 1:1 = 1024x1024)")
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS,
                        help=f"Number of inference steps (default: {NUM_INFERENCE_STEPS})")
    parser.add_argument("--negative-prompt", type=str, default=None,
                        help="Negative prompt. Only takes effect together with --cfg > 1 (~2.2x slower).")
    parser.add_argument("--cfg", type=float, default=TRUE_CFG_SCALE,
                        help=f"true_cfg_scale (default: {TRUE_CFG_SCALE} = off). Needs --negative-prompt.")
    parser.add_argument("--no-kv-cache", action="store_true",
                        help="Disable the prefix KV cache (2.6x slower on edits)")
    parser.add_argument("--output-dir", type=str, default=str(output_dir), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    width, height = ASPECT_RATIOS[args.aspect]
    use_kv_cache = not args.no_kv_cache

    # Load reference images once (they are reused for every prompt in the loop).
    ref_images = None
    if args.image:
        paths = [Path(p).expanduser() for p in args.image]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Reference image(s) not found: {missing}")
        if len(paths) > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"{len(paths)} reference images given but the model supports at most "
                f"{MAX_REFERENCE_IMAGES}."
            )
        ref_images = [load_image(str(p)).convert("RGB") for p in paths]

    pipe = QwenImage21Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)

    if ref_images:
        print(f"Reference images ({len(ref_images)}, in reading order):")
        for p in args.image:
            print(f"  {p}")
        print("Order matters: block-causal attention reads them in the order given,")
        print("and they are tagged <image1>, <image2>, ... so the prompt can refer to them.")
    else:
        print("Text-to-image mode (no reference images).")

    print(f"Qwen-Image-2.1 INT4 interactive generation. Type 'quit' or Ctrl+C to exit.")
    print(f"Output: {width}x{height}, {args.steps} steps, cfg={args.cfg}, kv_cache={use_kv_cache}")
    print("Seed: -1 or empty = random, otherwise a fixed integer seed.")
    if args.cfg > 1.0 and not args.negative_prompt:
        print("WARNING: --cfg > 1 without --negative-prompt is ignored (diffusers warns and skips CFG).")
    if args.negative_prompt and args.cfg <= 1.0:
        print("WARNING: --negative-prompt with cfg <= 1 is ignored (diffusers warns and skips CFG).")
    counter = 0

    while True:
        try:
            prompt = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            print("bye.")
            break

        try:
            seed_input = input("seed> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if seed_input.lower() in {"quit", "exit", "q"}:
            print("bye.")
            break

        # -1 or empty -> random seed; otherwise parse a fixed integer seed
        if seed_input in {"", "-1"}:
            seed = torch.seed() % (2**32)
        else:
            try:
                seed = int(seed_input) % (2**32)
            except ValueError:
                print(f"invalid seed {seed_input!r}, using a random seed instead")
                seed = torch.seed() % (2**32)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=args.steps,
            true_cfg_scale=args.cfg,
            use_kv_cache=use_kv_cache,
            generator=generator,
        )
        if args.negative_prompt:
            kwargs["negative_prompt"] = args.negative_prompt
        if ref_images is not None:
            kwargs["image"] = ref_images

        image = pipe(**kwargs).images[0]

        counter += 1
        output_path = out_dir / f"qwen_image_21_int4_{counter:04d}_seed{seed}.png"
        image.save(output_path)
        print(f"saved {output_path}  (seed={seed})")


if __name__ == "__main__":
    main()
