# ViT-Adapter for Video Semantic Segmentation

Applying the **[ViT-Adapter](https://github.com/czczup/ViT-Adapter)** dense-prediction
model to **Video Semantic Segmentation (VSS)** — running a Cityscapes-trained
`Mask2Former + BEiT-Adapter-L` segmentor frame-by-frame over real dashcam footage and
stitching the colour-coded predictions back into a video.

This repository documents a hands-on experiment as part of my research direction in
**computer vision**.

---

## 🎯 Research Direction

My research interest is in **visual scene understanding** — teaching machines to parse
the world at the pixel level. The threads I am actively exploring:

- **Semantic & dense prediction** — per-pixel labelling of images and video (semantic,
  instance, and panoptic segmentation).
- **Vision Transformers for dense tasks** — how plain ViT backbones, with lightweight
  *adapters* such as ViT-Adapter, can rival purpose-built hierarchical backbones
  (Swin, ConvNeXt) on detection and segmentation.
- **Perception for autonomous driving** — street-scene understanding on benchmarks like
  **Cityscapes**: roads, vehicles, pedestrians, traffic signs, and other safety-critical
  classes.
- **From images to video** — extending strong single-image models to the temporal domain
  (video semantic segmentation), and studying the speed/quality trade-offs that matter for
  real deployment.

This project is a concrete step along that path: take a state-of-the-art image
segmentation model, understand its architecture and inference pipeline end-to-end, and
adapt it to a video task on my own data.

---

## 🧪 What I Did

1. **Selected the model.** ViT-Adapter (Chen *et al.*, ICLR 2023) — an adapter that
   injects image-specific inductive biases into a plain ViT via spatial prior, feature
   interaction, and injector modules, enabling strong dense prediction.
2. **Downloaded the pretrained checkpoint.** `Mask2Former + BEiT-Adapter-L`, trained on
   **Cityscapes** (19 classes, **84.9 mIoU**, ~2.2 GB).
3. **Built a video-inference pipeline.** Since ViT-Adapter is an *image* model, VSS is done
   by running the segmentor on **each frame** and re-encoding the overlay into an `.mp4`
   (`infer_video.py`).
4. **Applied it to my own sample video** — a 30-second dashcam clip — and produced a
   colour-segmented result.
5. **Ran everything on Google Colab (free T4 GPU)**, fully reproducibly, in
   [`ViT_Adapter_VSS_Colab.ipynb`](ViT_Adapter_VSS_Colab.ipynb).

---

## 🎬 Result

Side-by-side comparison — **left: original dashcam frame · right: ViT-Adapter semantic
segmentation** (Cityscapes palette, first 10 seconds):

https://github.com/kainatzahra98/ViT-Adapter/raw/main/results/comparison.mp4

> If the player doesn't load inline, the clip is here:
> [`results/comparison.mp4`](results/comparison.mp4)

Each colour is a Cityscapes class — e.g. road, sidewalk, car, person, vegetation, sky,
building, traffic sign — predicted independently for every frame.

---

## 🗂️ Repository Contents

| File | Description |
|---|---|
| [`ViT_Adapter_VSS_Colab.ipynb`](ViT_Adapter_VSS_Colab.ipynb) | End-to-end Colab notebook: environment setup, model download, inference, and visualisation. Runs top-to-bottom on a free T4. |
| [`infer_video.py`](infer_video.py) | Frame-by-frame video segmentation script (config/checkpoint/video in, segmented `.mp4` out) with speed/quality knobs. |
| [`results/comparison.mp4`](results/comparison.mp4) | The original-vs-segmented comparison video. |

---

## 🚀 How to Reproduce

The whole pipeline runs on **Google Colab** (no local GPU needed):

1. Open `ViT_Adapter_VSS_Colab.ipynb` in Colab and set the runtime to **GPU (T4)**.
2. Run the cells top-to-bottom — they create the environment, compile the custom
   deformable-attention CUDA op, download the Cityscapes checkpoint, and write the
   inference script.
3. Upload your own video and run the segmentation cell.

### The inference script

```bash
python infer_video.py \
  --config configs/cityscapes/mask2former_beit_adapter_large_896_80k_cityscapes_ss.py \
  --checkpoint /content/ckpt/model.pth \
  --video /content/input.mp4 \
  --out   /content/output_seg.mp4 \
  --scale 1600x896 \
  --seconds 10 \
  --stride 3
```

**Key options**

| Flag | Meaning |
|---|---|
| `--scale WxH` | Model input size. This checkpoint uses a fixed 896 window, so the **short side must be ≥ 896** (`1600x896` is efficient; `2048x1024` is full quality). |
| `--seconds N` | Process only the first *N* seconds of the video. |
| `--stride N` | Run every *N*-th frame (trades smoothness for speed). |
| `--opacity` | Overlay strength of the segmentation mask. |

---

## 🛠️ Technical Stack

- **Model:** ViT-Adapter (BEiT-Adapter-L backbone) + Mask2Former head
- **Framework:** PyTorch 1.10 + OpenMMLab (MMSegmentation 0.20.2 / MMCV-full 1.4.2 / MMDetection 2.22.0)
- **Dataset / weights:** Cityscapes pretrained checkpoint
- **Hardware:** Google Colab T4 GPU

---

## 🙏 Acknowledgements

This work builds directly on the original **[ViT-Adapter](https://github.com/czczup/ViT-Adapter)**
by Chen *et al.* All credit for the model and pretrained weights goes to the original authors.

> Chen, Z., Duan, Y., Wang, W., He, J., Lu, T., Dai, J., & Qiao, Y. (2023).
> *Vision Transformer Adapter for Dense Predictions.* ICLR 2023.
