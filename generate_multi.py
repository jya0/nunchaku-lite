"""Z-Image Turbo INT4 (rank 256) multi-GPU batched prompt generation.

Checkpoint:
    https://huggingface.co/nunchaku-ai/nunchaku-z-image-turbo/blob/main/svdq-int4_r256-z-image-turbo.safetensors

Run from the repository root:

    python generate_multi.py

Workflow:
  1. Optionally load one or more LoRAs (with per-adapter strength).
  2. Enter the path to a text file containing one prompt per line,
     or type prompts directly (empty line / 'done' to generate).
  3. Each prompt generates IMAGES_PER_PROMPT images in a single batched
     denoising pass.

Multi-GPU: when more than one CUDA GPU is available, the script spawns one
persistent worker process per GPU. Each worker loads its own pipeline copy and
processes prompts from a shared queue, giving near-linear speedup.

All images are saved under outputs/ with their seed in the filename.
"""

# --- Must set CUDA_VISIBLE_DEVICES before importing torch ---
# When the parent process sets _NUNCHAKU_WORKER_GPU, this module is being
# re-imported by a spawned child worker. We immediately mask the process to
# that single GPU so that native CUDA kernels (which default to device 0)
# target the correct physical GPU.
import os

_WORKER_GPU_ENV = "_NUNCHAKU_WORKER_GPU"
if _WORKER_GPU_ENV in os.environ:
    # Re-export CUDA_VISIBLE_DEVICES for this child process only.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ[_WORKER_GPU_ENV]

import multiprocessing as mp
import queue
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from diffusers import ZImagePipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "Tongyi-MAI/Z-Image-Turbo"
checkpoint = "nunchaku-ai/nunchaku-z-image-turbo/svdq-int4_r256-z-image-turbo.safetensors"
precision = "int4"
output_dir = Path("outputs")

# Fixed generation settings (per request)
HEIGHT = 1280
WIDTH = 1280
NUM_INFERENCE_STEPS = 8
GUIDANCE_SCALE = 0.0

# Number of images generated per prompt in a single batched forward pass.
IMAGES_PER_PROMPT = 2


def read_prompts(path: str) -> list[str]:
    """Read a prompt file with one prompt per line, skipping blanks/comments."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"prompt file not found: {p}")
    prompts: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    return prompts


# --------------------------------------------------------------------------- #
# Worker process: loads a pipeline on one GPU and processes prompts forever.  #
# --------------------------------------------------------------------------- #

_WORKER_READY = "__READY__"
_WORKER_DONE = "__DONE__"
_SENTINEL = "__STOP__"


def gpu_worker(
    gpu_id: int,
    lora_specs: list[tuple[str, float]],
    task_queue: "queue.Queue",
    result_queue: "queue.Queue",
    run_dir: str,
) -> None:
    """Run on a single GPU, pulling prompts from ``task_queue`` until sentinel.

    The parent process sets the ``_NUNCHAKU_WORKER_GPU`` env var before
    spawning each child, which causes the module-level code (running before
    ``import torch``) to set ``CUDA_VISIBLE_DEVICES`` accordingly. Inside the
    worker, we always use ``cuda:0`` — which maps to the single visible device.

    Args:
        gpu_id: Physical CUDA device index for this worker.
        lora_specs: List of ``(path, strength)`` LoRAs to load.
        task_queue: Queue of ``(prompt_idx, prompt, base_seed, img_start)``
            tuples. A prompt_idx of -1 signals shutdown.
        result_queue: Queue for status/error messages back to the main process.
        run_dir: Output directory path for saved images.
    """
    device = "cuda:0"  # always 0 within this process (masked by CUDA_VISIBLE_DEVICES)
    try:
        pipe = load_nunchaku_pipeline(
            model_id,
            pipeline_cls=ZImagePipeline,
            checkpoint=checkpoint,
            target="z_image",
            precision=precision,
            torch_dtype=torch.bfloat16,
            device=device,
        )
        pipe = pipe.to(device)

        # Load LoRAs
        loaded_adapters: list[str] = []
        loaded_weights: list[float] = []
        for lora_path, strength in lora_specs:
            adapter_name = Path(lora_path).stem
            if adapter_name in loaded_adapters:
                continue
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            loaded_adapters.append(adapter_name)
            loaded_weights.append(strength)
        if loaded_adapters:
            pipe.set_adapters(loaded_adapters, adapter_weights=loaded_weights)

        result_queue.put((_WORKER_READY, gpu_id))
        run_dir_path = Path(run_dir)

        with torch.inference_mode():
            while True:
                try:
                    item = task_queue.get(timeout=5)
                except queue.Empty:
                    continue
                if item[0] == _SENTINEL:
                    break

                prompt_idx, prompt, base_seed, img_start = item

                # Generate one image at a time (no batching) to reduce peak VRAM.
                seeds: list[int] = []
                cur_seed = base_seed
                t0 = time.perf_counter()
                for img_offset in range(IMAGES_PER_PROMPT):
                    s = cur_seed if cur_seed is not None else torch.seed() % (2**32)
                    cur_seed = None if cur_seed is None else (cur_seed + 1) % (2**32)
                    seeds.append(s)
                    generator = torch.Generator(device=device).manual_seed(s)

                    image = pipe(
                        prompt=prompt,
                        height=HEIGHT,
                        width=WIDTH,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        num_images_per_prompt=1,
                        generator=generator,
                    ).images[0]

                    img_num = img_start + img_offset
                    fname = f"{prompt_idx:03d}_{img_num}_seed{s}.png"
                    output_path = run_dir_path / fname
                    image.save(output_path)
                    del image
                    torch.cuda.empty_cache()

                dt = time.perf_counter() - t0

                result_queue.put(("done", (gpu_id, prompt_idx, dt, seeds)))

        result_queue.put((_WORKER_DONE, gpu_id))
    except Exception as exc:  # noqa: BLE001
        result_queue.put(("error", (gpu_id, repr(exc))))


# --------------------------------------------------------------------------- #
# Main orchestration                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs detected.")

    print(f"Detected {num_gpus} GPU(s):")
    for i in range(num_gpus):
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  cuda:{i} - {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")

    use_multi = num_gpus > 1
    if use_multi:
        resp = input(f"\nUse all {num_gpus} GPUs in parallel? [Y/n]> ").strip().lower()
        use_multi = resp not in {"n", "no"}

    if use_multi:
        gpu_ids = list(range(num_gpus))
        print(f"Spawning {num_gpus} worker processes (one pipeline per GPU).")
    else:
        gpu_id = 0
        if num_gpus > 1:
            while True:
                choice = input(f"select GPU [0-{num_gpus - 1}] (default 0)> ").strip()
                if not choice:
                    break
                try:
                    gpu_id = int(choice)
                    if 0 <= gpu_id < num_gpus:
                        break
                except ValueError:
                    pass
                print(f"invalid choice {choice!r}, try again")
        gpu_ids = [gpu_id]
        print(f"Single-GPU mode on cuda:{gpu_id}.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- LoRA loading (paths only; workers load weights themselves) ---
    lora_specs: list[tuple[str, float]] = []
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
        lora_specs.append((lora_path, strength))
        print(f"Added LoRA: {lora_path} (strength={strength})")

    if lora_specs:
        summary = ", ".join(f"{Path(p).stem}={w}" for p, w in lora_specs)
        print(f"LoRAs to load: {summary}")
    else:
        print("No LoRAs.")

    # --- Prompt file or manual entry ---
    prompts_file = input("\npath to prompt file (empty for manual entry)> ").strip()
    from_file: bool = False
    file_prompts: list[str] | None = None
    if prompts_file:
        try:
            file_prompts = read_prompts(prompts_file)
        except FileNotFoundError as exc:
            print(f"{exc}  Falling back to manual entry.")
        else:
            if file_prompts:
                from_file = True
            else:
                print("prompt file is empty. Falling back to manual entry.")
                file_prompts = None

    # --- Run subfolder ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if lora_specs:
        lora_combo = "_".join(f"{Path(p).stem}{w}" for p, w in lora_specs)
    else:
        lora_combo = "nolora"
    lora_combo = re.sub(r"[^A-Za-z0-9._-]", "_", lora_combo)
    run_dir = output_dir / f"{timestamp}_{lora_combo}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {run_dir}")

    # --- Spawn workers ---
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    workers = []
    for gid in gpu_ids:
        # Set env var in the parent's actual environment so the spawned child
        # inherits it at fork/exec time. The module-level code (top of file)
        # reads this before importing torch.
        os.environ[_WORKER_GPU_ENV] = str(gid)
        p = ctx.Process(
            target=gpu_worker,
            args=(gid, lora_specs, task_queue, result_queue, str(run_dir)),
            daemon=True,
        )
        p.start()
        workers.append(p)
    # Clean up the env var in the parent so it doesn't leak
    os.environ.pop(_WORKER_GPU_ENV, None)

    # Wait for all workers to report ready
    ready_gpus = set()
    errors: list[str] = []
    timeout = 600  # 10 min for model loading
    deadline = time.perf_counter() + timeout
    while len(ready_gpus) < len(gpu_ids) and time.perf_counter() < deadline:
        try:
            tag, payload = result_queue.get(timeout=5)
        except queue.Empty:
            continue
        if tag == _WORKER_READY:
            ready_gpus.add(payload)
            print(f"  worker on cuda:{payload} ready ({len(ready_gpus)}/{len(gpu_ids)})")
        elif tag == "error":
            gid, msg = payload
            print(f"  worker on cuda:{gid} FAILED: {msg}")
            errors.append(msg)

    if errors or len(ready_gpus) < len(gpu_ids):
        print("Not all workers started. Aborting.")
        for _ in gpu_ids:
            task_queue.put((_SENTINEL, "", None, 0))
        for p in workers:
            p.join(timeout=5)
        return

    print(f"\nAll {len(gpu_ids)} worker(s) ready. Generating {IMAGES_PER_PROMPT} images per prompt.")

    # --- Generation loop ---
    try:
        while True:
            if from_file and file_prompts is not None:
                prompts = list(file_prompts)
                print(f"\nLoaded {len(prompts)} prompt(s) from {prompts_file}.")
            else:
                print("\nEnter prompts, one per line. Empty line or 'done' to generate; 'quit' to exit.")
                prompts = []
                while True:
                    try:
                        line = input("prompt> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        line = "quit"
                    if line.lower() in {"quit", "exit", "q"}:
                        break
                    if line.lower() in {"done", ""}:
                        break
                    prompts.append(line)

                if line.lower() in {"quit", "exit", "q"}:
                    print("bye.")
                    break
                if not prompts:
                    print("no prompts entered.")
                    continue

            seed_input = input("base seed (empty/-1 for random)> ").strip()
            base_seed: int | None = None
            if seed_input and seed_input not in {"-1"}:
                try:
                    base_seed = int(seed_input) % (2**32)
                except ValueError:
                    print(f"invalid seed {seed_input!r}, using random seeds")

            total = len(prompts)
            print(
                f"Generating {IMAGES_PER_PROMPT} images for each of {total} prompt(s) "
                f"({IMAGES_PER_PROMPT * total} total) across {len(gpu_ids)} GPU(s)... Ctrl+C to abort."
            )

            # Distribute prompts round-robin across workers
            total_start = time.perf_counter()
            queued = 0
            for idx, prompt in enumerate(prompts, start=1):
                task_queue.put((idx, prompt, base_seed, 1))
                queued += 1

            # Collect results
            completed = 0
            while completed < queued:
                try:
                    tag, payload = result_queue.get(timeout=600)
                except queue.Empty:
                    print("Timeout waiting for worker results.")
                    break
                if tag == "done":
                    gid, prompt_idx, dt, seeds = payload
                    completed += 1
                    print(
                        f"  [gpu:{gid}] prompt {prompt_idx}/{total} done in {dt:.2f}s "
                        f"(seeds: {seeds})"
                    )
                elif tag == "error":
                    gid, msg = payload
                    print(f"  [gpu:{gid}] ERROR: {msg}")
                    completed += 1

            total_dt = time.perf_counter() - total_start
            total_images = IMAGES_PER_PROMPT * total
            print(
                f"\nRun done. {total_images} images saved to {run_dir}/ "
                f"in {total_dt:.1f}s ({total_images / total_dt:.2f} img/s)."
            )

            if from_file:
                break
            print("You can keep entering more prompts. Type 'quit' to exit.")

    except KeyboardInterrupt:
        print("\nAborted by user.")
    finally:
        # Shut down workers
        for _ in gpu_ids:
            task_queue.put((_SENTINEL, "", None, 0))
        for p in workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        print("Workers shut down.")


if __name__ == "__main__":
    main()
