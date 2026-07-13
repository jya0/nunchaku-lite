"""Test whether untested LoRAs can be loaded by Z-Image Turbo.

Loads the Z-Image Turbo INT4 pipeline once, then attempts to load each LoRA
and reports success/failure. No image generation is performed — this only
validates that the LoRA weights are compatible with the pipeline's
``load_lora_weights`` path.
"""

from pathlib import Path

import torch
from diffusers import ZImagePipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "Tongyi-MAI/Z-Image-Turbo"
checkpoint = "nunchaku-ai/nunchaku-z-image-turbo/svdq-int4_r256-z-image-turbo.safetensors"
precision = "int4"

LORA_FILES = [
    "loras/ZITnsfwLoRAv2.safetensors",
    "loras/ZITnsfwLoRAv3.safetensors",
    "loras/NSFW_master_ZIT_000017532.safetensors",
]


def main() -> None:
    print("Loading Z-Image Turbo INT4 pipeline...")
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
    print("Pipeline loaded successfully.\n")

    all_ok = True
    for lora_path in LORA_FILES:
        lora_path = str(Path(lora_path).expanduser())
        adapter_name = Path(lora_path).stem
        print(f"--- Testing: {lora_path} ---")
        if not Path(lora_path).is_file():
            print(f"  ✗ File not found: {lora_path}\n")
            all_ok = False
            continue
        try:
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            pipe.set_adapters([adapter_name], adapter_weights=[1.0])
            print(f"  ✓ Loaded successfully (adapter='{adapter_name}', strength=1.0)")
            # Clean up: delete adapter so the next test starts fresh
            pipe.delete_adapters([adapter_name])
            print(f"  ✓ Adapter '{adapter_name}' deleted cleanly")
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")
            all_ok = False
        print()

    print("=" * 60)
    if all_ok:
        print("RESULT: All LoRAs loaded successfully ✓")
    else:
        print("RESULT: Some LoRAs failed to load ✗")
    print("=" * 60)


if __name__ == "__main__":
    main()
