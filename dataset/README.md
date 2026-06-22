# Dataset Preparation

SCISSR is trained on **EndoVis 2018** and evaluated zero-shot (out-of-distribution)
on **CholecSeg8k**. Neither dataset is redistributed here — download them from the
official sources and arrange them as described below. All paths are relative to the
repository root (run the scripts from the repo root).

## 1. EndoVis 2018 (in-distribution, training + test)

Source: MICCAI 2018 Robotic Scene Segmentation Challenge
(<https://endovissub2018-roboticscenesegmentation.grand-challenge.org/>).

Expected layout:

```
dataset/Endovision18/
├── raw/
│   ├── Train_Data/
│   │   ├── seq_1/
│   │   │   ├── left_frames/      # *.png RGB frames
│   │   │   └── labels/           # *.png RGB semantic masks
│   │   ├── seq_2/
│   │   └── ...
│   └── Test_Data/
│       ├── seq_1/
│       └── ...
└── train_val_split.json          # generated, see below
```

Generate the train/val split used in the paper:

```bash
python train/prepare_endovis18_split.py
```

This writes `dataset/Endovision18/train_val_split.json` (sequence-level 80/20 split
designed to balance the foreground class distribution).

## 2. CholecSeg8k (out-of-distribution, test only)

Source: <https://www.kaggle.com/datasets/newslab/cholecseg8k>.
No CholecSeg8k data is used during training — it is only used to evaluate
cross-domain generalization.

Expected layout:

```
dataset/CholecSeg8k/
├── video01/
│   ├── video01_00080/
│   │   ├── frame_80_endo.png
│   │   └── frame_80_endo_watershed_mask.png
│   └── ...
└── ...
```

## Class definitions

EndoVis 2018 foreground class IDs / colors are defined inline in
`train/train_stage2.py` and `train/prepare_endovis18_split.py`. CholecSeg8k class
handling lives in `eval/eval_cholecseg8k.py`.
