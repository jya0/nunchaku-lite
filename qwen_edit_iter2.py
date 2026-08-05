#!/usr/bin/env python3
"""Qwen-Image-Edit-2509 INT4 — targeted iteration-2 anatomy correction.

Chained edit: takes the iteration-1 outputs as input, applies a PER-IMAGE
targeted prompt to fix only the remaining problems (extra digits, extra limbs,
disconnected parts, missing genitals).

Loads the model ONCE, then processes each image with its own specific prompt
and CFG. Uses the proven Two-Part Formula (targeted action + preservation lock).
"""

import time
from pathlib import Path

import torch
from diffusers import QwenImageEditPlusPipeline

from qwen_image_edit import DEFAULT_NEGATIVE_PROMPT, load_pipeline, edit_image

# ---------------------------------------------------------------------------
# Iteration-2 task map: <input image stem> -> (prompt, cfg)
#   Inputs are the iter1 outputs (chained edit), so disfigurements already
#   fixed remain untouched and only the listed problem is corrected.
# ---------------------------------------------------------------------------
INPUT_DIR = Path("outputs/anatomy_iter1")
OUTPUT_DIR = Path("outputs/anatomy_iter2")

ITER2_TASKS = {
    # --- Extra digits / limbs (removal) ---
    "004_2_seed1296160615_edited_s42_cfg2.0.png": (
        "Remove the extra toe. Fix the feet so each foot has exactly five natural toes, "
        "evenly spaced with proper proportions. Remove the sixth toe on the foot. "
        "Keep everything else in the image unchanged.",
        2.0,
    ),
    "009_2_seed3284485612_edited_s42_cfg2.0.png": (
        "Remove the extra third hand. Render exactly two natural hands, each with five "
        "well-proportioned fingers and natural joints. Remove the additional hand. "
        "Keep everything else unchanged.",
        3.0,
    ),
    "032_2_seed3192197747_edited_s42_cfg2.0.png": (
        "Fix the disfigured hand. Render a natural, correctly proportioned hand with "
        "exactly five fingers and natural joints. Correct the deformed fingers. "
        "Keep everything else unchanged.",
        3.0,
    ),
    "035_2_seed2308378209_edited_s42_cfg2.0.png": (
        "Remove the extra foot and the disfigured limb in the middle of the body. "
        "Render a normal, anatomically correct body with two arms and two legs. "
        "Keep everything else unchanged.",
        3.0,
    ),
    "036_2_seed3256783737_edited_s42_cfg2.0.png": (
        "Reconnect the arm to the shoulder. Fix the arm so it is properly attached "
        "with natural proportions and no gap. Keep everything else unchanged.",
        2.0,
    ),
    "046_2_seed2771211976_edited_s42_cfg2.0.png": (
        "Remove the extra leg. Fix the body so there are exactly two natural legs, "
        "correctly proportioned and attached to the hips. Remove the additional leg. "
        "Keep everything else unchanged.",
        3.0,
    ),

    # --- Severe blob reconstruction ---
    "060_2_seed4091667936_edited_s42_cfg2.0.png": (
        "Reconstruct the entire body with natural, anatomically correct proportions. "
        "Fix the deformed blob of flesh into a coherent human body with proper limbs, "
        "torso, and features. Keep the background and lighting unchanged.",
        3.0,
    ),

    # --- Genital restoration (body fix removed them) ---
    "009_1_seed4046682410_edited_s42_cfg2.0.png": (
        "Restore the genitals. Render natural, clearly visible female genitals "
        "(vulva) in the correct location, uncensored and unaltered. "
        "Keep everything else unchanged.",
        2.0,
    ),
    "043_2_seed4121256306_edited_s42_cfg2.0.png": (
        "Restore the genitals for both people in the scene. Render fully visible, "
        "natural female vulva and male penis in the correct anatomical locations, "
        "uncensored. Keep everything else unchanged.",
        2.0,
    ),
    "044_1_seed2325802616_edited_s42_cfg2.0.png": (
        "Restore the genitals. Render natural, fully visible female genitals (vulva) "
        "in the correct location, uncensored. Keep everything else unchanged.",
        2.0,
    ),
    "044_2_seed3265451053_edited_s42_cfg2.0.png": (
        "Restore the genitals. Render natural, fully visible female genitals (vulva) "
        "in the correct location, uncensored. Keep everything else unchanged.",
        2.0,
    ),
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    tasks = []
    for stem, (prompt, cfg) in ITER2_TASKS.items():
        src = INPUT_DIR / stem
        if not src.exists():
            print(f"!! MISSING input: {src}", flush=True)
            continue
        tasks.append((src, prompt, cfg))
    print(f"\nIteration 2: {len(tasks)} targeted edits\n", flush=True)

    pipe: QwenImageEditPlusPipeline = load_pipeline(128)

    for idx, (src, prompt, cfg) in enumerate(tasks, 1):
        print(f"\n[{idx}/{len(tasks)}] --- {src.name} (cfg={cfg}) ---", flush=True)
        t0 = time.time()
        try:
            edit_image(
                pipe,
                str(src),
                prompt=prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                steps=8,
                true_cfg_scale=cfg,
                seed=42,
                width=None,
                height=None,
                output_dir=OUTPUT_DIR,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR on {src.name}: {e}", flush=True)
        print(f"  [{idx}/{len(tasks)}] done in {time.time()-t0:.1f}s", flush=True)

    print("\nIteration 2 complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())