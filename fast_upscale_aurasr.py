#!/usr/bin/env python3
"""
Fast Image Upscaler — AuraSR-v2 (CodeFormer optional)

Non-diffusion upscaling pipeline:
  1. AuraSR-v2: GAN-based 4x super-resolution (fast, high quality)
  2. CodeFormer: Face restoration (optional, w=0.7 balance between quality & fidelity)

Expected speed: ~1-3s per image on RTX 4090

Usage:
    .venv/bin/python fast_upscale_aurasr.py <input_folder> <output_folder> [options]

Examples:
    # Default: AuraSR 4x only (no face restoration)
    .venv/bin/python fast_upscale_aurasr.py ./inputs ./outputs

    # 2x upscale instead of 4x
    .venv/bin/python fast_upscale_aurasr.py ./inputs ./outputs --upscale 2

    # Enable CodeFormer face restoration
    .venv/bin/python fast_upscale_aurasr.py ./inputs ./outputs --codeformer

    # CodeFormer weight (0.0=fidelity, 1.0=quality)
    .venv/bin/python fast_upscale_aurasr.py ./inputs ./outputs --codeformer --codeformer-w 0.5
"""

import argparse
import os
import sys
import time
import glob
import traceback

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast Image Upscaler — AuraSR-v2 + CodeFormer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input folder containing images")
    parser.add_argument("output", help="Output folder for upscaled images")
    parser.add_argument("--upscale", type=int, default=4, help="Upscale factor (default: 4, AuraSR native)")
    parser.add_argument("--no-codeformer", action="store_true", default=True,
                        help="Skip CodeFormer face restoration (AuraSR only, default: True)")
    parser.add_argument("--codeformer", action="store_true", default=False,
                        help="Enable CodeFormer face restoration")
    parser.add_argument("--codeformer-w", type=float, default=0.7,
                        help="CodeFormer fidelity weight: 0.0=fidelity, 1.0=quality (default: 0.7)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id (default: 0)")
    parser.add_argument("--half", action="store_true", default=False,
                        help="Use fp16 for AuraSR (faster, slightly lower quality)")
    parser.add_argument("--tile-size", type=int, default=128,
                        help="AuraSR tile size for processing (default: 128)")
    parser.add_argument("--overlap", type=int, default=16,
                        help="Tile overlap in pixels (default: 16)")
    parser.add_argument("--max-tiles", type=int, default=0,
                        help="Max tiles per image (0=unlimited, for testing)")
    return parser.parse_args()


def load_aurasr(device=0, half=False):
    """Load AuraSR-v2 model via spandrel."""
    from spandrel import ModelLoader
    from safetensors.torch import load_file

    model_path = os.path.join(SCRIPT_DIR, "thirdparty", "aurasr", "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"AuraSR model not found at {model_path}")

    print(f"  Loading AuraSR-v2 from {model_path}...")
    state = load_file(model_path)
    loader = ModelLoader()
    model = loader.load_from_state_dict(state).model

    device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    model = model.to(device_str)
    if half:
        model = model.half()
    model.eval()
    print(f"  AuraSR loaded: {type(model).__name__}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, device_str


def load_codeformer(device=0):
    """Load CodeFormer model for face restoration."""
    from basicsr.archs.codeformer_arch import CodeFormer

    model_path = os.path.join(SCRIPT_DIR, "thirdparty", "codeformer", "codeformer-v0.1.0.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CodeFormer model not found at {model_path}")

    print(f"  Loading CodeFormer from {model_path}...")
    net = CodeFormer(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=['32', '64', '128', '256']
    )
    ckpt = torch.load(model_path, map_location='cpu')['params_ema']
    net.load_state_dict(ckpt)
    device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    net = net.to(device_str).eval()
    print(f"  CodeFormer loaded: params={sum(p.numel() for p in net.parameters())/1e6:.1f}M")
    return net, device_str


def setup_face_helper(device=0):
    """Set up face detection helper.

    FaceRestoreHelper auto-downloads detection/parsing models on first use.
    We pre-download them to the expected location to avoid issues.
    """
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    from basicsr.utils.download_util import load_file_from_url

    codeformer_dir = os.path.join(SCRIPT_DIR, "thirdparty", "codeformer")

    # Pre-place detection model in the location facelib expects
    # facelib looks in: <basicsr_install>/weights/facelib/
    import basicsr
    basicsr_dir = os.path.dirname(basicsr.__file__)
    weights_dir = os.path.join(os.path.dirname(basicsr_dir), "weights", "facelib")
    os.makedirs(weights_dir, exist_ok=True)

    for src_name, dst_name in [
        ("detection_Resnet50_Final.pth", "detection_Resnet50_Final.pth"),
        ("parsing_parsenet.pth", "parsing_parsenet.pth"),
    ]:
        src = os.path.join(codeformer_dir, src_name)
        dst = os.path.join(weights_dir, dst_name)
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)
            print(f"  Placed {dst_name} in facelib weights dir")

    face_helper = FaceRestoreHelper(
        upscale_factor=1,  # We handle upscaling separately
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        device=f"cuda:{device}" if torch.cuda.is_available() else "cpu",
    )
    print(f"  Face detection model loaded")
    return face_helper


@torch.no_grad()
def upscale_aurasr(model, img_tensor, device, tile_size=128, overlap=16, half=False):
    """
    Upscale image with AuraSR using tiled processing.
    img_tensor: (1, C, H, W) in [0, 1] range
    Returns: (1, C, H*4, W*4) in [0, 1] range
    """
    _, C, H, W = img_tensor.shape
    dtype = torch.float16 if half else torch.float32
    img_tensor = img_tensor.to(device, dtype=dtype)

    # If image is small enough, process directly
    if H <= tile_size and W <= tile_size:
        output = model(img_tensor)
        return output

    # Tiled processing
    scale = 4  # AuraSR is a 4x upscaler
    stride = tile_size - overlap
    output_h, output_w = H * scale, W * scale
    output = torch.zeros(1, C, output_h, output_w, device=device, dtype=dtype)
    weight = torch.zeros(1, 1, output_h, output_w, device=device, dtype=dtype)

    # Generate tile positions
    h_positions = list(range(0, max(H - tile_size, 0) + 1, stride))
    if h_positions[-1] + tile_size < H:
        h_positions.append(max(H - tile_size, 0))
    w_positions = list(range(0, max(W - tile_size, 0) + 1, stride))
    if w_positions[-1] + tile_size < W:
        w_positions.append(max(W - tile_size, 0))

    if not h_positions:
        h_positions = [0]
    if not w_positions:
        w_positions = [0]

    for y in h_positions:
        for x in w_positions:
            # Extract tile
            tile = img_tensor[:, :, y:y+tile_size, x:x+tile_size]
            if tile.shape[2] < tile_size or tile.shape[3] < tile_size:
                # Pad to tile_size
                pad_h = tile_size - tile.shape[2]
                pad_w = tile_size - tile.shape[3]
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')

            # Process tile
            tile_out = model(tile)

            # Crop to actual tile output size
            actual_h = min(tile_size, H - y) * scale
            actual_w = min(tile_size, W - x) * scale
            tile_out = tile_out[:, :, :actual_h, :actual_w]

            # Create weight mask (smooth fade at edges)
            mask = torch.ones(1, 1, actual_h, actual_w, device=device, dtype=dtype)
            if overlap > 0:
                fade = min(overlap, actual_h // 2, actual_w // 2)
                if fade > 1:
                    fade_mask = torch.linspace(0, 1, fade, device=device, dtype=dtype)
                    fade_mask_rev = torch.flip(fade_mask, dims=[0])
                    mask[0, 0, :fade, :] *= fade_mask[:, None]
                    mask[0, 0, -fade:, :] *= fade_mask_rev[:, None]
                    mask[0, 0, :, :fade] *= fade_mask[None, :]
                    mask[0, 0, :, -fade:] *= fade_mask_rev[None, :]

            out_y = y * scale
            out_x = x * scale
            output[:, :, out_y:out_y+actual_h, out_x:out_x+actual_w] += tile_out * mask
            weight[:, :, out_y:out_y+actual_h, out_x:out_x+actual_w] += mask

    output = output / (weight + 1e-8)
    return output


@torch.no_grad()
def restore_faces(codeformer_net, face_helper, img_bgr, w=0.7, device='cuda:0'):
    """
    Restore faces in image using CodeFormer.
    img_bgr: numpy array (H, W, 3) in BGR order, uint8
    Returns: restored image (H, W, 3) BGR uint8
    """
    from torchvision.transforms.functional import normalize

    face_helper.clean_all()
    face_helper.read_image(img_bgr)
    face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    face_helper.align_warp_face()

    if len(face_helper.cropped_faces) == 0:
        return img_bgr  # No faces detected

    # Restore each face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        cropped_face_t = torch.from_numpy(cropped_face / 255.).float()
        cropped_face_t = cropped_face_t.permute(2, 0, 1).unsqueeze(0)  # BGR -> (1,3,H,W)
        # BGR to RGB
        cropped_face_t = cropped_face_t.flip(1)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.to(device)

        try:
            output = codeformer_net(cropped_face_t, w=w, adain=True)[0]
            restored_face = output.squeeze(0).flip(0)  # RGB -> BGR
            restored_face = (restored_face.clamp(-1, 1) + 1) / 2 * 255
            restored_face = restored_face.permute(1, 2, 0).cpu().numpy().astype('uint8')
        except Exception as e:
            print(f"    CodeFormer failed on face {idx}: {e}")
            restored_face = cropped_face

        face_helper.add_restored_face(restored_face)

    # Paste faces back
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()
    return restored_img


def process_image(img_path, output_path, aurasr_model, codeformer_net, face_helper, args, device):
    """Process a single image."""
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)  # (H, W, 3) RGB

    # Convert to tensor (1, 3, H, W) in [0, 1]
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    # Step 1: AuraSR upscale
    upscaled = upscale_aurasr(
        aurasr_model, img_tensor, device,
        tile_size=args.tile_size, overlap=args.overlap, half=args.half
    )

    # Handle non-4x scale
    if args.upscale != 4:
        _, _, H, W = upscaled.shape
        target_h = img_tensor.shape[2] * args.upscale
        target_w = img_tensor.shape[3] * args.upscale
        upscaled = F.interpolate(upscaled, size=(target_h, target_w), mode='bilinear', align_corners=False)

    # Convert to numpy BGR for CodeFormer/OpenCV
    upscaled_np = upscaled.squeeze(0).clamp(0, 1).cpu().numpy()
    upscaled_np = (upscaled_np * 255).astype('uint8')
    upscaled_np = upscaled_np.transpose(1, 2, 0)  # (H, W, 3) RGB
    upscaled_bgr = cv2.cvtColor(upscaled_np, cv2.COLOR_RGB2BGR)

    # Step 2: CodeFormer face restoration (disabled by default)
    if args.codeformer and codeformer_net is not None:
        upscaled_bgr = restore_faces(
            codeformer_net, face_helper, upscaled_bgr,
            w=args.codeformer_w, device=device
        )

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, upscaled_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    return upscaled_bgr.shape[1], upscaled_bgr.shape[0]


def main():
    args = parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # Find images
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp')
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(args.input, ext)))
        img_files.extend(glob.glob(os.path.join(args.input, '**', ext), recursive=True))
    img_files = sorted(set(img_files))

    if not img_files:
        print(f"ERROR: No images found in {args.input}", file=sys.stderr)
        sys.exit(1)

    # Print banner
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Fast Upscaler — AuraSR-v2 + CodeFormer                 ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Input:      {args.input:<44}║")
    print(f"║  Output:     {args.output:<44}║")
    print(f"║  Images:     {len(img_files):<44}║")
    print(f"║  Upscale:    {str(args.upscale)+'x':<44}║")
    cf_status = f"enabled (w={args.codeformer_w})" if args.codeformer else "disabled"
    print(f"║  CodeFormer: {cf_status:<44}║")
    print(f"║  Tile size:  {str(args.tile_size)+'px':<44}║")
    print(f"║  Precision:  {'fp16' if args.half else 'fp32':<44}║")
    print(f"║  Device:     {device:<44}║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Load models
    print("Loading models...")
    t0 = time.time()
    aurasr_model, device = load_aurasr(args.device, args.half)

    codeformer_net = None
    face_helper = None
    if args.codeformer:
        codeformer_net, _ = load_codeformer(args.device)
        face_helper = setup_face_helper(args.device)

    torch.cuda.empty_cache()
    print(f"  Models loaded in {time.time()-t0:.1f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
    print()

    # Process images
    os.makedirs(args.output, exist_ok=True)
    total_time = 0
    processed = 0
    failed = 0

    for idx, img_path in enumerate(img_files):
        rel_path = os.path.relpath(img_path, args.input)
        output_path = os.path.join(args.output, os.path.splitext(rel_path)[0] + '.png')

        print(f"[{idx+1}/{len(img_files)}] {rel_path}")

        t0 = time.time()
        try:
            w, h = process_image(
                img_path, output_path, aurasr_model,
                codeformer_net, face_helper, args, device
            )
            elapsed = time.time() - t0
            total_time += elapsed
            processed += 1
            print(f"  → {w}x{h} in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED ({elapsed:.2f}s): {e}")
            traceback.print_exc()
            failed += 1

        if args.max_tiles > 0 and processed >= args.max_tiles:
            print(f"  Reached --max-tiles limit ({args.max_tiles})")
            break

    # Summary
    print()
    print("═" * 58)
    print(f"  Processed: {processed}/{len(img_files)}")
    print(f"  Failed:    {failed}")
    if processed > 0:
        avg_time = total_time / processed
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Avg/image:  {avg_time:.2f}s")
        print(f"  Throughput: {1/avg_time:.1f} img/s")
    print(f"  VRAM peak:  {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("═" * 58)

    # Save summary
    summary_path = os.path.join(args.output, "_benchmark.txt")
    with open(summary_path, 'w') as f:
        f.write(f"AuraSR-v2 Benchmark\n")
        f.write(f"{'='*50}\n")
        f.write(f"Images: {processed}/{len(img_files)}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        if processed > 0:
            f.write(f"Avg/image: {total_time/processed:.2f}s\n")
            f.write(f"Throughput: {1/(total_time/processed):.1f} img/s\n")
        f.write(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB\n")
        f.write(f"Settings: upscale={args.upscale}x, tile={args.tile_size}, "
                f"overlap={args.overlap}, half={args.half}\n")
        f.write(f"CodeFormer: {cf_status}\n")


if __name__ == "__main__":
    main()
