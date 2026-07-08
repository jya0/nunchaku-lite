"""Z-Image Turbo INT4 (rank 256) interactive generation loop.

Checkpoint:
    https://huggingface.co/nunchaku-ai/nunchaku-z-image-turbo/blob/main/svdq-int4_r256-z-image-turbo.safetensors

Run from the repository root:

    python generate.py

Type a prompt at each `prompt>` line and press Enter to generate an image.
Then enter a seed at `seed>`: `-1` (or empty) for random, or a specific integer
to reproduce a fixed result. Type `quit` (or Ctrl+C) to exit.
All images are saved under outputs/ with their seed in the filename.
"""

from pathlib import Path

import torch
from diffusers import ZImagePipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "Tongyi-MAI/Z-Image-Turbo"
checkpoint = "nunchaku-ai/nunchaku-z-image-turbo/svdq-int4_r256-z-image-turbo.safetensors"
precision = "int4"
output_dir = Path("outputs")

# Fixed generation settings (per request)
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 8
GUIDANCE_SCALE = 0.0


def main() -> None:
    pipe = load_nunchaku_pipeline(
        model_id,
        pipeline_cls=ZImagePipeline,
        checkpoint=checkpoint,
        target="z_image",
        precision=precision,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    pipe = pipe.to("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prompt for one or more LoRAs with per-adapter strength
    loaded_adapters: list[str] = []
    loaded_weights: list[float] = []
    while True:
        lora_input = input("\nlora path (empty to finish)> ").strip()
        if not lora_input:
            break
        lora_path = str(Path(lora_input).expanduser())
        if not Path(lora_path).is_file():
            print(f"file not found: {lora_path}")
            continue
        strength_input = input("strength (empty for 1.0)> ").strip()
        try:
            strength = float(strength_input) if strength_input else 1.0
        except ValueError:
            print(f"invalid strength {strength_input!r}, using 1.0")
            strength = 1.0
        adapter_name = Path(lora_path).stem
        if adapter_name in loaded_adapters:
            print(f"adapter {adapter_name!r} already loaded, skipping")
            continue
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        loaded_adapters.append(adapter_name)
        loaded_weights.append(strength)
        # Activate all loaded LoRAs together with their per-adapter weights
        pipe.set_adapters(loaded_adapters, adapter_weights=loaded_weights)
        print(f"Loaded LoRA: {lora_path} (strength={strength})")

    if loaded_adapters:
        summary = ", ".join(f"{n}={w}" for n, w in zip(loaded_adapters, loaded_weights))
        print(f"Active LoRAs: {summary}")
    else:
        print("No LoRA loaded.")

    print("Z-Image Turbo INT4 r256 interactive generation. Type 'quit' or Ctrl+C to exit.")
    print("Seed: -1 or empty = random, otherwise a fixed integer seed.")
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

        image = pipe(
            prompt=prompt,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).images[0]

        counter += 1
        output_path = output_dir / f"z_image_turbo_int4_r256_{counter:04d}_seed{seed}.png"
        image.save(output_path)
        print(f"saved {output_path}  (seed={seed})")


if __name__ == "__main__":
    main()
