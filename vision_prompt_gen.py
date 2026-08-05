#!/usr/bin/env python3
"""Vision-LLM guided anatomy prompt generator.

Phase 1 of the two-phase anatomy-correction pipeline:
  Phase 1 (THIS): a vision-capable LLM inspects each image, identifies defects,
    writes a targeted edit prompt per defect, AND returns a normalized bounding
    box per defect (for masked/region editing).
  Phase 2: Qwen-Image-Edit-2511 (--fast 4-step + runtime NSFW LoRA) executes the
      prompts, optionally using the boxes as crop.

Uses vLLM's structured output (json_schema guided decoding) so the model returns
valid JSON: {"defects": [{"part", "description", "fix", "box"}, ...]}. Guided
decoding keeps fields reliable and minimizes tokens. When nothing is wrong, the
defects array is empty.

The vision model and the edit model cannot be loaded on the same GPU at once, so
this runs as a SEPARATE phase that ONLY calls the vision endpoint (no edit model
loaded). It writes a JSON manifest of {image_path -> {prompt, cfg, crop, defects}}
for phase 2.

Also writes DEBUG overlays: the input image with each defect box drawn and the
exact prompt text that will be fed to Qwen-Image-Edit, so you can audit what the
pipeline decided.
"""

import argparse
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw

# Vision model (uncensored Qwen3.6-27B with vision), served by vLLM locally.
DEFAULT_ENDPOINT = "http://localhost:8889/v1"
DEFAULT_MODEL = "thinkingcap-abliterated-nvfp4"

# Structured output schema (guided decoding) — keeps fields reliable, fewer tokens.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "defects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "part": {"type": "string"},
                    "description": {"type": "string"},
                    "fix": {"type": "string"},
                    "box": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["part", "description", "fix", "box"],
            },
        }
    },
    "required": ["defects"],
}

SYSTEM_PROMPT = (
    "You are a vision inspector for adult/NSFW images. You look at a nude human figure and "
    "decide whether any body part is anatomically wrong and needs fixing by an image editor.\n\n"
    "Describe ONLY the parts that are actually wrong. Be concrete and specific about what is "
    "wrong: wrong digit count (six toes, four fingers), a missing or detached limb/digit, fused "
    "or melted digits, a hand/foot merged into the body, a clearly wrong object (e.g. a shoe on "
    "an otherwise bare foot). For each defect give:\n"
    "- part: the body part name, with the subject's own left/right.\n"
    "- description: exactly what is wrong, stated as a fact. NO hedging ('looks', 'appears', "
    "  'maybe', 'seems').\n"
    "- fix: the desired end result to change to (e.g. 'exactly five separated toes', 'left foot "
    "  bare to match the other').\n"
    "- box: the tight bounding box of the defective region, normalized 0..1, top-left origin "
    "  (x=0 left, y=0 top), as [x1, y1, x2, y2].\n\n"
    "Do NOT invent problems: a normal pose (hand behind the head, a foreshortened foot, fingers "
    "overlapping in perspective, a limb bent, a hand merely touching the body) is NOT a defect. "
    "When in doubt the digit/limb is probably just overlapping another — do not report it.\n\n"
    "If everything looks anatomically normal, return {\"defects\": []}.\n\n"
    "You must respond ONLY with valid JSON matching the required schema."
)


def _client(endpoint: str) -> OpenAI:
    return OpenAI(api_key="EMPTY", base_url=endpoint)


def encode_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def inspect_image(
    image_path: str,
    endpoint: str,
    model: str,
    thinking: bool = True,
) -> dict:
    """Call the vision LLM with structured JSON output and return parsed JSON.

    No max_tokens is passed so the model always finishes thinking and emits its
    structured `content` (a JSON object). Guided decoding guarantees valid JSON.
    """
    client = _client(endpoint)
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
                    }
                ],
            },
        ],
        # Qwen official best-practice sampling params for THINKING mode.
        # CRITICAL: do NOT use temperature=0 (greedy) here — Qwen docs explicitly
        # warn it causes performance degradation and ENDLESS REPETITION loops in
        # the chain-of-thought. Use temp 0.6 + top_p 0.95 + top_k 20 + min_p 0,
        # plus a small presence_penalty to suppress repetition.
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": thinking},
            "top_k": 20,
            "min_p": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "defects", "schema": OUTPUT_SCHEMA},
            },
        },
    }
    try:
        resp = client.chat.completions.create(**kwargs)
    finally:
        client.close()
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        return {"defects": []}
    # Strip any markdown code fences if present.
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"defects": []}


def build_edit_prompt(defect: dict) -> str:
    """Turn one defect dict into a natural-language edit instruction."""
    part = (defect.get("part") or "the defect").strip()
    desc = (defect.get("description") or "").strip()
    fix = (defect.get("fix") or "").strip()
    sentence = f"Fix the {part}"
    if desc:
        sentence += f": {desc}"
    if fix:
        sentence += f". Make it so {fix}."
    return sentence


def boxes_to_crops(defects: list[dict], width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Convert normalized [x1,y1,x2,y2] boxes to pixel (x,y,w,h) crop tuples."""
    crops = []
    for d in defects:
        box = d.get("box")
        if not box or len(box) < 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        px = round(x1 * width)
        py = round(y1 * height)
        pw = round((x2 - x1) * width)
        ph = round((y2 - y1) * height)
        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))
        pw = max(16, min(pw, width - px))
        ph = max(16, min(ph, height - py))
        crops.append((px, py, pw, ph))
    return crops


def draw_debug_overlay(src: Path, defects: list[dict], dest: Path) -> None:
    """Draw each defect box + its edit prompt onto a copy of the image.

    Produces an audit image showing exactly what the pipeline decided and what
    prompt will be sent to Qwen-Image-Edit.
    """
    img = Image.open(src).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, h // 50))
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    colors = ["#FF2D55", "#00C7B2", "#FFB000", "#7B4DFF", "#FF5C00", "#00B06A"]
    for i, d in enumerate(defects):
        box = d.get("box")
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        px1, py1 = round(x1 * w), round(y1 * h)
        px2, py2 = round(x2 * w), round(y2 * h)
        col = colors[i % len(colors)]
        draw.rectangle([px1, py1, px2, py2], outline=col, width=max(3, h // 200))
        label = (d.get("part") or "defect").strip()
        ty = max(0, py1 - int(h * 0.03))
        draw.text((px1 + 2, ty), label, fill=col, font=font)
    # Bottom band with the prompts that will be fed to the edit model.
    prompts = [build_edit_prompt(d) for d in defects]
    if prompts:
        lines = [f"[{i + 1}] {p}" for i, p in enumerate(prompts)]
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        line_h = max(20, int(h * 0.022))
        band_h = line_h * (len(lines) + 1) + 8
        odraw.rectangle([0, h - band_h, w, h], fill=(0, 0, 0, 175))
        y = h - band_h + 5
        for ln in lines:
            odraw.text((10, y), ln, fill=(255, 255, 255), font=font)
            y += line_h
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Vision-LLM targeted anatomy prompt generator")
    ap.add_argument("--input-dir", type=str, required=True, help="Dir of images to inspect")
    ap.add_argument("--output", type=str, required=True, help="Output JSON manifest path")
    ap.add_argument("--debug-dir", type=str, default=None,
                    help="Dir for debug overlays (defect boxes + edit prompt drawn on image)")
    ap.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--no-thinking", action="store_true",
                    help="Disable thinking (enable_thinking:false) for a direct answer.")
    ap.add_argument("--ext", nargs="+", default=[".png", ".jpg", ".jpeg", ".webp", ".bmp"])
    ap.add_argument("--select", type=str, default=None,
                    help="Comma-separated subset of filenames (optional, for testing)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of concurrent requests (default 4). The vLLM server can "
                         "batch concurrent requests, so this massively speeds up batches.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    images = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in args.ext)
    if args.select:
        want = set(args.select.split(","))
        images = [f for f in images if f.name in want or f.stem in want]

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Inspecting {len(images)} images with {args.model} "
          f"(thinking={'off' if args.no_thinking else 'on'}, workers={args.workers}, "
          f"json_schema guided, no token cap)...\n", flush=True)

    total = len(images)
    done = 0
    manifest: dict[str, dict] = {}

    def _run(img: Path):
        try:
            data = inspect_image(img, args.endpoint, args.model, thinking=not args.no_thinking)
            defects = data.get("defects", []) if isinstance(data, dict) else []
            with Image.open(img) as im:
                w, h = im.size
            prompts = [build_edit_prompt(d) for d in defects]
            crops = boxes_to_crops(defects, w, h)
            entry = {"prompt": "", "cfg": 1.0, "defects": defects, "prompts": prompts}
            if prompts:
                entry["prompt"] = prompts[0]
            if crops:
                entry["crop"] = crops[0]
                if len(crops) > 1:
                    entry["crops"] = crops
            return str(img), entry, None
        except Exception as e:  # noqa: BLE001
            return str(img), None, f"{e}"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run, img): img for img in images}
        for fut in as_completed(futs):
            done += 1
            img_key, entry, err = fut.result()
            if err:
                print(f"[{done}/{total}] {Path(img_key).name} -> ERROR: {err}", flush=True)
                continue
            manifest[img_key] = entry
            nd = len(entry.get("defects", []))
            label = "NONE" if nd == 0 else "; ".join(p[:70] for p in entry["prompts"])
            print(f"[{done}/{total}] {Path(img_key).name} -> {label}", flush=True)
            if args.debug_dir and nd > 0:
                draw_debug_overlay(Path(img_key), entry["defects"],
                                   Path(args.debug_dir) / Path(img_key).name)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    n_def = sum(1 for v in manifest.values() if v.get("defects"))
    print(f"\nSaved manifest ({len(manifest)} entries, {n_def} with defects) -> {args.output}",
          flush=True)


if __name__ == "__main__":
    main()