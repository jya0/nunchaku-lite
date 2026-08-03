# Best Model for Each Task

This page summarizes the best-performing models for each image generation/editing task, based on hands-on testing on an NVIDIA RTX 4090 (48GB, Ada Lovelace sm_89) using nunchaku-lite INT4 quantization.

## Text-to-Image: Z-Image Turbo

**Model:** Z-Image Turbo

Z-Image Turbo is the best model for text-to-image generation. It produces high-quality images with fast inference thanks to its distilled architecture. Used with nunchaku-lite INT4 quantization, it runs efficiently on consumer Ada GPUs.

Key strengths:
- Fast generation (turbo-distilled, fewer steps needed)
- High image quality
- Works well with LoRA fine-tuning

## Upscaling: SeedVR2 7B

**Model:** SeedVR2 7B

SeedVR2 7B is the best model for image upscaling. It provides high-fidelity super-resolution with sharp detail reconstruction.

Key strengths:
- High-quality upscaling with fine detail recovery
- 7B parameter model — balances quality and speed
- Preserves identity and composition of the source image

## Anatomy Correction / Image Editing: Qwen Image Edit

**Model:** Qwen Image Edit

Qwen Image Edit is the current best model for correcting anatomy issues in generated images (extra limbs, deformed hands/feet, fused fingers, extra digits). It outperforms alternatives like FLUX.2 Klein for this specific task.

Key strengths:
- Reliable anatomy correction (extra limbs, deformed hands, fused fingers)
- Instruction-based editing — describe what to fix and what to keep
- Preserves identity, face, pose, and background while correcting target areas

> **Note:** FLUX.2 Klein was tested for anatomy correction but removed — it could not reliably fix anatomy issues (turned feet into hands, failed to remove extra limbs). See commit history for details.
