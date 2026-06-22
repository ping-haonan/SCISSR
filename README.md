# SCISSR: Scribble-Conditioned Interactive Surgical Segmentation and Refinement

Official implementation of **SCISSR** (MICCAI 2026).

SCISSR equips **SAM 2** with **scribble-conditioned, multi-round interactive
refinement** for surgical scene segmentation. Instead of sparse points or coarse
boxes, the annotator traces the target with freehand scribbles; the model turns
these strokes into dense prompt embeddings and iteratively refines the mask as the
user draws corrective strokes on error regions.

SCISSR achieves **95.41% Dice on EndoVis 2018** (5 rounds) and **96.30% Dice on the
out-of-distribution CholecSeg8k** (3 rounds), outperforming iterative point
prompting on both benchmarks.

> The architecture is intentionally backbone-agnostic: the Scribble Encoder, Spatial
> Gated Fusion (SGF), and LoRA adapters attach only through standard embedding
> interfaces, so the same components transfer to other prompt-driven segmentation
> architectures without structural changes.

---

## Method overview

Three lightweight, trainable components are added on top of a **frozen** SAM 2 image
encoder:

1. **Scribble Encoder** — converts a two-channel (positive / negative) scribble map
   into a dense prompt embedding aligned with SAM 2's image-embedding resolution.
2. **Spatial Gated Fusion (SGF)** — injects only the *latest* correction scribble
   into the Memory Attention query, with a learnable gate (`alpha`, zero-initialized,
   so SGF starts as identity).
3. **Toggleable LoRA adapters** — inserted into the query/value projections of the
   mask decoder and Memory Attention; zero-initialized so training starts from the
   pretrained behavior.

A **dual-track** scribble pathway drives multi-round refinement:
- **Track 1 (accumulated → dense prompt):** the union of all scribbles is encoded and
  added to the mask decoder's dense prompt, preserving the full interaction history.
- **Track 2 (latest → memory query):** only the current round's scribble is fused
  into Memory Attention via SGF, focusing attention on newly corrected regions.

Scribbles are synthesized from masks by an **Adaptive Scribble Generator**
(centerline / wave-skeleton / contour / line) — see
`scissr/interactions/`.

---

## Repository layout

```
SCISSR/
├── sam2/                          # [third-party] vendored SAM 2 backbone + Hydra configs (Meta AI)
├── scissr/                        # SCISSR components (our code)
│   ├── models/                    # ScribbleSam2Memory (incl. SGF), LoRA, Scribble Encoder
│   └── interactions/              # adaptive / correction scribble generation
├── train/                         # training scripts
│   ├── train_stage1.py            # Stage 1 (Scribble Encoder + decoder LoRA)
│   ├── train_stage2.py            # Stage 2 (full iterative pipeline)
│   └── prepare_endovis18_split.py # build the EndoVis 2018 train/val split
├── eval/                          # evaluation scripts
│   ├── eval_endovis18.py          # in-distribution evaluation
│   └── eval_cholecseg8k.py        # out-of-distribution evaluation
├── checkpoints/
│   ├── download_sam2.sh           # downloads the frozen SAM 2.1 Tiny backbone
│   └── scissr/                    # SCISSR's lightweight trained weights (shipped)
│       ├── stage1_best.pt         # Stage 1 (Scribble Encoder + decoder LoRA)
│       └── stage2_best.pt         # Stage 2 (full pipeline) — use this for inference
└── dataset/                       # data preparation guide (data not redistributed)
```

The shipped SCISSR weights are tiny (a few MB) because only the added modules are
trained — the SAM 2 backbone stays frozen.

> **Note on `sam2/`.** This directory is **third-party code vendored from
> [facebookresearch/sam2](https://github.com/facebookresearch/sam2)** (Apache-2.0,
> © Meta Platforms, Inc.), included here only so the repo runs out of the box. It is
> **not** part of our contribution. All SCISSR code lives under `scissr/`, `train/`,
> and `eval/`. See [`NOTICE`](NOTICE) for full third-party attribution.

---

## Installation

```bash
conda create -n scissr python=3.10 -y
conda activate scissr

# Install a torch/torchvision build matching your CUDA version first, e.g.:
# pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Download the frozen SAM 2.1 Tiny backbone:

```bash
bash checkpoints/download_sam2.sh   # -> checkpoints/sam2.1_hiera_tiny.pt
```

> Run all scripts from the repository root so the relative default paths resolve.

---

## Data

See [`dataset/README.md`](dataset/README.md) for preparing **EndoVis 2018**
(training + in-distribution test) and **CholecSeg8k** (out-of-distribution test).
The datasets are not redistributed here.

---

## Evaluation (with the released weights)

EndoVis 2018 (in-distribution):

```bash
python eval/eval_endovis18.py \
    --model_path checkpoints/scissr/stage2_best.pt \
    --ckpt_path  checkpoints/sam2.1_hiera_tiny.pt \
    --test_dir   dataset/Endovision18/raw/Test_Data \
    --num_rounds 5
```

CholecSeg8k (out-of-distribution, zero-shot):

```bash
python eval/eval_cholecseg8k.py \
    --model_path checkpoints/scissr/stage2_best.pt \
    --ckpt_path  checkpoints/sam2.1_hiera_tiny.pt \
    --data_root  dataset/CholecSeg8k \
    --num_rounds 3 --no_sam3
```

> `eval_cholecseg8k.py` can optionally compare against a SAM 3 baseline, which
> requires the (non-public) `sam3` package and checkpoint. Pass `--no_sam3` (as
> above) to evaluate SCISSR only — no extra dependency needed.

---

## Training

SCISSR is trained in **two stages** on EndoVis 2018 (single RTX 4090, 24 GB).

First build the EndoVis 2018 train/val split:

```bash
python train/prepare_endovis18_split.py   # -> dataset/Endovision18/train_val_split.json
```

**Stage 1** — Scribble Encoder + mask-decoder LoRA (no memory / SGF):

```bash
python train/train_stage1.py \
    --data_dir dataset/Endovision18/raw/Train_Data \
    --ckpt_path checkpoints/sam2.1_hiera_tiny.pt \
    --only_lora --epochs 5
# -> trained_models/lora_comparison/<run>/with_lora/best_model.pt
```

**Stage 2** — full iterative pipeline (Scribble Encoder + SGF + decoder LoRA +
Memory Attention LoRA), progressively annealing from box to pure scribble prompts:

```bash
python train/train_stage2.py \
    --data_dir dataset/Endovision18/raw/Train_Data \
    --ckpt_path checkpoints/sam2.1_hiera_tiny.pt \
    --stage1_ckpt checkpoints/scissr/stage1_best.pt \
    --epochs 10 --base_lr 1e-4 --finetune_lr 1e-5 --num_rounds 3
# -> trained_models/stage2_progressive/<run>/best_model.pt
```

The released `checkpoints/scissr/stage1_best.pt` and `stage2_best.pt` correspond to
these two stages, so you can reproduce evaluation without retraining.

---

## Results

| Method | EndoVis 2018 (ID) R4 mDice | CholecSeg8k (OOD) R2 mDice |
|---|---|---|
| SAM 2 Tiny (10 pt/ch) | 72.08 | 83.18 |
| SAM 2 Tiny (BBox) | 84.62 (R0) | 84.62 (R0) |
| **SCISSR (Contour)** | **95.41** | **96.30** |

See the paper for full per-round, per-class, and convergence results.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{scissr2026,
  title     = {SCISSR: Scribble-Conditioned Interactive Surgical Segmentation and Refinement},
  author    = {<authors>},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

> Update the author list and publication details for the camera-ready version.

---

## Acknowledgements

This project builds on excellent prior work:

- [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2) — Meta AI.
- [ScribblePrompt](https://github.com/halleewong/ScribblePrompt) — scribble-based
  interactive biomedical segmentation, which inspired parts of the scribble
  synthesis and interaction code.

See [`NOTICE`](NOTICE) for attribution details.

## License

Released under the **Apache License 2.0** — see [`LICENSE`](LICENSE).
The bundled `sam2/` code is also under Apache 2.0 (© Meta Platforms, Inc.).
Dataset terms are governed by their respective providers.
