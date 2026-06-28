"""
ViT-Adapter — Video Semantic Segmentation (frame-by-frame)
==========================================================
ViT-Adapter is an *image* semantic-segmentation model. This script applies it to
the Video Semantic Segmentation (VSS) task by running the segmentor on each frame
of a video and re-encoding the colourised overlay back into an .mp4.

Run it from inside the repo's `segmentation/` directory (so the custom
`mmcv_custom` / `mmseg_custom` packages are importable), e.g.:

    python infer_video.py \
        --config configs/cityscapes/mask2former_beit_adapter_large_896_80k_cityscapes_ss.py \
        --checkpoint /content/ckpt/model.pth \
        --video /content/input.mp4 \
        --out /content/output_seg.mp4 \
        --scale 1024x512
"""
import argparse
import os

# Force a headless matplotlib backend BEFORE mmseg imports pyplot. Colab exports
# MPLBACKEND=module://matplotlib_inline... which is invalid in a plain subprocess.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402
matplotlib.use("Agg")

import cv2  # noqa: E402
import mmcv  # noqa: E402
from mmcv import Config  # noqa: E402

# These two imports register the ViT-Adapter backbones / Mask2Former heads with
# the mmseg registry. They MUST be imported before building the segmentor.
import mmcv_custom   # noqa: F401,E402
import mmseg_custom  # noqa: F401,E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmseg.apis import inference_segmentor  # noqa: E402
from mmseg.models import build_segmentor  # noqa: E402
from mmseg.core.evaluation import get_classes, get_palette  # noqa: E402


def load_model(cfg, checkpoint_path, device, palette_name):
    """Build the segmentor and load weights, tolerating a checkpoint that has no
    `meta` block (this ViT-Adapter release ships a bare state_dict). We set
    CLASSES / PALETTE from the requested palette instead of from the checkpoint."""
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if cfg.model.get("backbone") is not None:
        cfg.model.backbone.pretrained = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = load_checkpoint(model, checkpoint_path, map_location="cpu")
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    model.CLASSES = meta.get("CLASSES", get_classes(palette_name))
    model.PALETTE = meta.get("PALETTE", get_palette(palette_name))
    model.cfg = cfg            # mmseg's inference_segmentor reads model.cfg
    model.to(device)
    model.eval()
    return model


def parse_scale(s):
    w, h = s.lower().split("x")
    return (int(w), int(h))


def main():
    ap = argparse.ArgumentParser(description="ViT-Adapter video semantic segmentation")
    ap.add_argument("--config", required=True, help="path to mmseg config .py")
    ap.add_argument("--checkpoint", required=True, help="path to the .pth weights")
    ap.add_argument("--video", required=True, help="input video file")
    ap.add_argument("--out", default="output_seg.mp4", help="output video file")
    ap.add_argument("--palette", default="cityscapes",
                    help="colour palette: cityscapes | ade20k | cocostuff")
    ap.add_argument("--opacity", type=float, default=0.5,
                    help="mask overlay opacity (0=video only, 1=mask only)")
    ap.add_argument("--scale", type=parse_scale, default=(1600, 896),
                    help="model input scale WxH. NOTE: this checkpoint has a fixed "
                         "896 window, so the SHORT side must be >= 896 or it "
                         "crashes (tensor 1793 vs 3137). 2048x1024 = full quality.")
    ap.add_argument("--stride", type=int, default=1,
                    help="run on every Nth frame (2 = half the frames, faster)")
    ap.add_argument("--seconds", type=float, default=0,
                    help="process only the first N seconds of the video "
                         "(0 = whole video). Output keeps real-time speed.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after this many processed frames (0 = whole video)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # Build the config; the pretrain paths are cleared inside load_model() so we
    # don't need the separate BEiT pretrain file at inference time.
    cfg = Config.fromfile(args.config)
    # Override the test-time resize so it fits in GPU memory / runs faster.
    for t in cfg.data.test.pipeline:
        if t.get("type") == "MultiScaleFlipAug":
            t["img_scale"] = args.scale

    print(f"Loading model on {args.device} (input scale {args.scale}) ...")
    model = load_model(cfg, args.checkpoint, args.device, args.palette)
    palette = get_palette(args.palette)

    reader = mmcv.VideoReader(args.video)
    src_fps = reader.fps or 25
    out_fps = src_fps / max(1, args.stride)
    # Limit to the first N seconds of *source* video, if requested.
    src_limit = int(args.seconds * src_fps) if args.seconds else 0
    span = f", first {args.seconds:g}s ({src_limit} frames)" if src_limit else ""
    print(f"Video: {reader.width}x{reader.height} @ {src_fps:.2f}fps, "
          f"{len(reader)} frames{span} -> writing {args.out} @ {out_fps:.2f}fps")

    writer = None
    n = 0
    for idx, frame in enumerate(mmcv.track_iter_progress(reader)):
        if src_limit and idx >= src_limit:
            break
        if idx % args.stride != 0:
            continue
        result = inference_segmentor(model, frame)
        vis = model.show_result(frame, result, palette=palette,
                                opacity=args.opacity, show=False)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(
                args.out, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
        writer.write(vis)
        n += 1
        if args.max_frames and n >= args.max_frames:
            break

    if writer is not None:
        writer.release()
    print(f"\n[done] wrote {n} segmented frames -> {args.out}")


if __name__ == "__main__":
    main()
