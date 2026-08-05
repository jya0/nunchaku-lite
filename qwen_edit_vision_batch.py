#!/usr/bin/env python3
"""Phase 2 of the vision-guided anatomy pipeline.

Reads the manifest JSON produced by `vision_prompt_gen.py` (phase 1) and runs
Qwen-Image-Edit (nunchaku INT4, --fast 4-step, runtime NSFW LoRA) on each image
using the vision-LLM-written targeted prompt.

Images whose vision prompt == "NONE" (or missing) are skipped.

Usage:
    python qwen_edit_vision_batch.py --manifest /tmp/vision_manifest.json \
        --output-dir outputs/vision_guided --fast
"""

import argparse
import json
from pathlib import Path

from qwen_image_edit import DEFAULT_NEGATIVE_PROMPT, load_pipeline, edit_image

# Per-image output may need multiple iterations of vision -> edit -> re-inspect.
# This script does the FIRST edit pass. Re-run after a fresh vision pass to refine.


def main() -> None:
    ap = argparse.ArgumentParser(description="Vision-guided edit batch (phase 2)")
    ap.add_argument("--manifest", type=str, required=True, help="JSON from vision_prompt_gen.py")
    ap.add_argument("--output-dir", type=str, default="outputs/vision_guided")
    ap.add_argument("--model", type=str, choices=["2509", "2511"], default="2511")
    ap.add_argument("--rank", type=str, choices=["32", "128", "256"], default="32")
    ap.add_argument("--fast", action="store_true", help="Use 4-step lightning mode")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--true-cfg-scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora", type=str, default=None)
    ap.add_argument("--lora-weight", type=float, default=0.1)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0, help="Skip first N entries (resume)")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for img_path, entry in manifest.items():
        prompt = (entry.get("prompt") or "").strip() if isinstance(entry, dict) else str(entry)
        if not prompt or prompt.upper() == "NONE":
            print(f"SKIP {Path(img_path).name} (NONE)", flush=True)
            continue
        if not Path(img_path).exists():
            print(f"!! MISSING input {img_path}", flush=True)
            continue
        cfg = float(entry.get("cfg", 1.0)) if isinstance(entry, dict) else 1.0
        # Optional spatial region (bounding box) for masked/region editing.
        crop = entry.get("crop") if isinstance(entry, dict) else None
        tasks.append((Path(img_path), prompt, cfg, crop))

    tasks = tasks[args.offset:]
    if args.max_images is not None:
        tasks = tasks[: args.max_images]

    if not tasks:
        print("No tasks to run (all NONE or missing).")
        return

    print(f"\nLoading edit model + running {len(tasks)} vision-guided edits (fast={args.fast})...\n")
    pipe = load_pipeline(args.model, args.rank, args.lora, args.lora_weight, args.fast)

    # Resolve steps/cfg defaults (mirror qwen_image_edit.main)
    steps = args.steps
    cfg = args.true_cfg_scale
    if args.fast and args.model == "2511":
        steps = steps or 4
        cfg = cfg if cfg is not None else 1.0
    elif args.model == "2509":
        steps = steps or 8
        cfg = cfg if cfg is not None else 2.0
    else:
        steps = steps or 40
        cfg = cfg if cfg is not None else 4.0

    ok = 0
    for i, (src, prompt, img_cfg, crop) in enumerate(tasks, 1):
        use_cfg = img_cfg if args.true_cfg_scale is None else args.true_cfg_scale
        crop_arg = f"{crop[0]},{crop[1]},{crop[2]},{crop[3]}" if crop else None
        print(f"\n[{i}/{len(tasks)}] --- {src.name} (cfg={use_cfg}{' crop=' + crop_arg if crop_arg else ''}) ---",
              flush=True)
        try:
            edit_image(
                pipe, str(src), prompt, DEFAULT_NEGATIVE_PROMPT,
                steps, use_cfg, args.seed,
                None, None, output_dir,
                crop=crop_arg,
            )
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR on {src.name}: {e}", flush=True)
    print(f"\nDone. {ok}/{len(tasks)} succeeded -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()