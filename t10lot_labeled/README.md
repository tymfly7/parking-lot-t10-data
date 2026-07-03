# Parking-Lot-T10 Dataset

A labeled, real-world parking-lot dataset captured from a single fixed camera at
**lot T10** Brno University of Technology. It ships in two ready-to-train forms:

- **Classification** — per-slot crops labeled `occupied` / `vacant`
- **Detection** — full frames in **YOLO** format with `vacant` / `occupied` boxes

Images were collected across many days and times of day (varying light, weather and
occupancy) and hand-organized into their occupancy labels. It is intended as a
compact, PKLot-style benchmark for training and evaluating parking-space occupancy
models on edge hardware (Raspberry Pi).

> This folder is the root of the standalone data repo
> **[parking-lot-t10-data](https://github.com/tymfly7/parking-lot-t10-data)**.
> It is kept out of the application repo (which ignores `backend/data/`) and pulled
> in separately — see [Getting the data](#getting-the-data).

---

## Contents at a glance

| Dataset | Task | Samples | Classes |
|---|---|---|---|
| `crops_classifiers/` | Classification (per slot) | **97,209** crops — 61,057 occupied · 36,152 vacant | `occupied`, `vacant` |
| `crops_yolo_detect/` | Detection (full frame) | **3,137** frames — 2,195 train · 470 val · 472 test | `vacant` (0), `occupied` (1) |

Both datasets are derived from the same source frames, so a detection frame and the
classification crops taken from it are consistent.

---

## Directory structure

```
t10lot_labeled/
├── crops_classifiers/            # ── classification dataset ──
│   ├── occupied/  *.jpg          #   61,057 cropped slots
│   └── vacant/    *.jpg          #   36,152 cropped slots
└── crops_yolo_detect/            # ── detection dataset (YOLO) ──
    ├── dataset.yaml              #   Ultralytics data config (nc=2)
    ├── images/{train,val,test}/  #   3,137 frames, split 2195/470/472
    └── labels/{train,val,test}/  #   one .txt per image (YOLO format)
```

---

## Classification dataset (`crops_classifiers/`)

Each file under `crops_classifiers/occupied/` or `crops_classifiers/vacant/` is a single
parking slot cropped from a full frame using its annotated polygon. The folder name
**is** the label.

```
crops_classifiers/
├── occupied/2026_02_20_..._roi01.jpg
└── vacant/2026_02_20_..._roi00.jpg
```

**Use with an ImageFolder-style loader:**

```python
from pathlib import Path
from torchvision import datasets, transforms

tf = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])
root = Path("~/parking-lot-t10/t10lot_labeled/crops_classifiers").expanduser()
ds = datasets.ImageFolder(root, transform=tf)
# ds.classes -> ['occupied', 'vacant']
```


---

## Detection dataset (`crops_yolo_detect/`)

Standard [Ultralytics YOLO](https://docs.ultralytics.com/datasets/detect/) layout:
each image has a matching `.txt` label file with one row per box:
`class cx cy w h` (normalized 0–1). Classes: `0 = vacant`, `1 = occupied`.

`dataset.yaml`:

```yaml
path: ~/parking-lot-t10/t10lot_labeled/crops_yolo_detect   # ~ expands to your home dir
train: images/train
val:   images/val
test:  images/test
nc: 2
names: {0: vacant, 1: occupied}
```

> **`path` is device-agnostic:** `~` expands to each user's home directory
> (Ultralytics calls `.expanduser()`), so it works on any machine where this repo is
> cloned under `~/parking-lot-t10`. If your checkout lives elsewhere, edit `path` to
> match.

**Train:**

```bash
yolo detect train model=yolo11n.pt data=~/parking-lot-t10/t10lot_labeled/crops_yolo_detect/dataset.yaml imgsz=640 epochs=100
```

The `crops_yolo_detect/` is regenerated from `detector_src/annotations.json` by
[`backend/src/data_prep/yolo_converter.py`](../../src/data_prep/yolo_converter.py)
(`build_yolo_detect_dataset`), so `detector_src/` is the source of truth and the YOLO
folder is reproducible.

---

## Annotation formats
- **`crops_yolo_detect/labels/**/*.txt`** — generated YOLO labels (one per image).

