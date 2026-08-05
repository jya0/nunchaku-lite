#!/usr/bin/env python3
"""Qwen-Image-Edit-2509 INT4 — targeted iteration-3 anatomy correction.

CRITICAL FIX vs iter2: iter2 re-edited iter1 outputs (chained edits), which
compounds information loss and instability. iter3 edits the ORIGINAL inputs
(tests/anatomy_test_iter1/) directly, so no quality is lost from prior passes.

Per-image targeted prompts (Two-Part Formula). Loads model ONCE.
"""

import time
from pathlib import Path

from qwen_image_edit import DEFAULT_NEGATIVE_PROMPT, load_pipeline, edit_image

# ---------------------------------------------------------------------------
# Iteration-3 task map: <original input stem in tests/anatomy_test_iter1/> -> prompt
#   Only images that still need fixing after iter2 (re-run from ORIGINAL).
#   (009_2, 032_2, 060_2 were fixed in iter2 -> not re-run.)
# ---------------------------------------------------------------------------
INPUT_DIR = Path("tests/anatomy_test_iter1")
OUTPUT_DIR = Path("outputs/anatomy_iter3")

ITER3_TASKS = {
    # 004_2: feet corrected from disfigured but still 6 toes; vulva got censored
    "004_2_seed1296160615.png": (
        "Fix the feet so each foot has exactly five natural toes, evenly spaced with "
        "proper proportions. Remove the sixth toe. Keep the face, pose, and "
        "everything else unchanged.",
        2.0,
    ),
    # 035_2: random foot removed, but vulva got censored
    "035_2_seed2308378209.png": (
        "Render the body with exactly two natural arms and two legs. Remove any extra "
        "foot or limb in the middle of the body. Keep everything else unchanged.",
        3.0,
    ),
    # 036_2: arm disconnected across the thigh when holding leg up
    "036_2_seed3256783737.png": (
        "Fix the arm so it is properly connected to the body at the shoulder with "
        "natural proportions and no gap or floating segment. The arm holding the leg "
        "should attach cleanly to the torso. Keep everything else unchanged.",
        3.0,
    ),
    # 046_2: extra leg removed but model added underwear/panties covering genitals
    "046_2_seed2771211976.png": (
        "Fix the body so there are exactly two natural legs correctly proportioned and "
        "attached to the hips. Remove the extra leg. Keep the woman nude with no "
        "underwear or clothing added. Keep everything else unchanged.",
        3.0,
    ),
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for stem, (prompt, cfg) in ITER3_TASKS.items():
        src = INPUT_DIR / stem
        if not src.exists():
            print(f"!! MISSING input: {src}", flush=True)
            continue
        tasks.append((src, prompt, cfg))
    print(f"\nIteration 3: {len(tasks)} targeted edits from ORIGINALS\n", flush=True)

    pipe = load_pipeline(128)

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

    print("\nIteration 3 complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())