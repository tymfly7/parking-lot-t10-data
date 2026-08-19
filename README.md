# Parking-Lot-T10 Dataset

A labeled, real-world parking-lot dataset captured from a single fixed camera
overlooking **parking lot T10** on the campus of **Brno University of Technology
(BUT)** in Brno, Czech Republic. Frames were collected across many days at a
10-minute period and across times of day (varying light, weather and occupancy)
and hand-organized into occupancy labels.

On top of these raw captures, this repository adds ready-to-train labels: the
parking bays are cropped and labeled in a PKLot-style layout for CNN classifiers,
and the full frames are annotated for YOLO object detection, then cleaned and
split for training.

The dataset ships in ready-to-train form:

- **Classification** - cropped images in PKLOT-style intended for **CNN** such as MobileNet format with `vacant` / `occupied`
  (`t10lot_labeled/crops_classifiers/`)
- **Bay occupancy detection** - full frames in **YOLO** format with one box per parking
  bay, labeled `vacant` / `occupied` (`t10lot_labeled/crops_yolo_detect/`)
- **Vehicle detection** - full frames in **YOLO** format with one box per vehicle, a
  single `vehicle` class (`t10lot_labeled/vehicle_yolo_detect/`)

- **Raw captures** - the original timestamped frames the labels are derived from
  (`DATA/`)

---

## Source & attribution

- **Site:** parking lot **T10**, Brno University of Technology (BUT), Brno, Czech Republic
- **Capture:** a single fixed Raspberry Pi camera on the BUT campus, one still roughly
  every 10 minutes, from early morning through the day into the evening, across a range
  of light, weather and occupancy (see [`PYTHON/camera.py`](PYTHON/camera.py))
- **Original dataset:** Kuzela (BUT thesis), captured on parking lot T10
- **Maintainer / expansion:** Tomáš Fryža (Brno University of Technology)
- **Labeling & ML packaging:** tymfly7 ([github](https://github.com/tymfly7))
- **License:** [MIT](LICENSE) - © 2026 Tomáš Fryža

If you use this dataset, please credit **Brno University of Technology** and the
original thesis by **Kuzela**, and link back to this repository.

---

## Contents at a glance

| Dataset | Task | Samples | Classes |
|---|---|---|---|
| `t10lot_labeled/crops_classifiers/` | Classification (per slot) | **97,209** crops - 61,057 occupied · 36,152 vacant | `occupied`, `vacant` |
| `t10lot_labeled/crops_yolo_detect/` | Bay occupancy detection (full frame) | **3,137** frames - 2,195 train · 470 val · 472 test | `vacant` (0), `occupied` (1) |
| `t10lot_labeled/vehicle_yolo_detect/` | Vehicle detection (full frame) | **190** frames - 149 train · 41 val | `vehicle` (0) |
| `DATA/` | Raw source frames | **3,336** timestamped `.jpg` across **48** capture days | — |

All three labeled datasets are derived from the same source captures held in `DATA/`, so a
detection frame and the classification crops taken from it are consistent.

The two detection datasets are labeled for different tasks. `crops_yolo_detect/` boxes
the parking bays and labels the state of each one. A single model trained on it returns
both the bay positions and the occupancy. `vehicle_yolo_detect/` boxes the vehicles and
carries no bay labels. Occupancy from it requires intersecting the boxes with a known
bay layout.

---

## Directory structure

```
parking-lot-t10-data/
├── DATA/                            # ── raw captures ──
│   └── YYYY-MM-DD/                  #   one folder per capture day
│       └── YYYY-MM-DD_HHMM.jpg      #   ~one still every 10 min
├── PYTHON/                          # ── capture tooling (Raspberry Pi) ──
│   ├── camera.py                    #   picamera2 still-capture loop
│   └── test_time.py
├── t10lot_labeled/                  # ── labeled datasets ──
│   ├── crops_classifiers/           #   classification dataset (per-slot crops)
│   │   ├── occupied/  *.jpg          #     61,057 cropped slots
│   │   └── vacant/    *.jpg          #     36,152 cropped slots
│   ├── crops_yolo_detect/           #   bay occupancy detection dataset (YOLO)
│   │   ├── dataset.yaml             #     Ultralytics data config (nc=2)
│   │   ├── images/{train,val,test}/ #     3,137 frames, split 2195/470/472
│   │   └── labels/{train,val,test}/ #     one .txt per image (YOLO format)
│   └── vehicle_yolo_detect/         #   vehicle detection dataset (YOLO)
│       ├── dataset.yaml             #     Ultralytics data config (nc=1)
│       ├── README.md                #     provenance, excluded frames, split
│       ├── images/{train,val}/      #     190 frames, split 149/41 by capture day
│       └── labels/{train,val}/      #     one .txt per image (YOLO format)
├── LICENSE
└── README.md
```

---

## Getting the data

Clone the repository (it is a standalone data repo, kept out of any application repo):

```bash
git clone https://github.com/tymfly7/parking-lot-t10-data ~/parking-lot-t10
```

Cloning under `~/parking-lot-t10` makes the paths in `dataset.yaml` (below) work
as-is. If your checkout lives elsewhere, edit `path` in `dataset.yaml` to match.

---

## Classification dataset (`t10lot_labeled/crops_classifiers/`)

Each file under `crops_classifiers/occupied/` or `crops_classifiers/vacant/` is a single
parking slot cropped from a full frame using its annotated polygon. The folder name
**is** the label - a PKLot-style layout ready for a `torchvision` `ImageFolder` loader
and CNN backbones such as MobileNet.

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

## Bay occupancy detection dataset (`t10lot_labeled/crops_yolo_detect/`)

Standard [Ultralytics YOLO](https://docs.ultralytics.com/datasets/detect/) layout:
each image has a matching `.txt` label file with one row per box:
`class cx cy w h` (normalized 0–1). Classes: `0 = vacant`, `1 = occupied`.

Each box covers a parking bay, not a vehicle. A model trained here proposes the bays
and classifies the state of each one in a single pass. That is an alternative to a
fixed bay layout combined with a per-crop classifier. Vehicles parked outside the
marked bays are not labeled and will not be detected. For vehicle boxes, use
`vehicle_yolo_detect/` below.

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

---

## Vehicle detection dataset (`t10lot_labeled/vehicle_yolo_detect/`)

The same Ultralytics layout with a single class, `0 = vehicle`. Each box covers a
vehicle anywhere in the frame. There are no bay boxes and no occupancy labels.

200 frames were sampled from `DATA/`, pre-labeled with stock COCO YOLO weights and
corrected by hand in [makesense.ai](https://www.makesense.ai/). Ten night frames were
dropped because the annotator could not see the vehicles in them. Six empty daylight
lots were kept with empty label files as negatives. That leaves 190 frames. The split
is by capture day, not by frame. No day contributes to both train and val. The
dataset's own
[`README.md`](t10lot_labeled/vehicle_yolo_detect/README.md) lists the excluded frames
and the day assignment in full.

`dataset.yaml`:

```yaml
path: ~/parking-lot-t10/t10lot_labeled/vehicle_yolo_detect   # ~ expands to your home dir
train: images/train
val: images/val
nc: 1
names: [vehicle]
```

**Train:**

```bash
yolo detect train model=yolo11n.pt data=~/parking-lot-t10/t10lot_labeled/vehicle_yolo_detect/dataset.yaml imgsz=640 epochs=100
```

---

## Raw captures (`DATA/`)

The `DATA/` tree holds the original frames straight from the camera, grouped into one
folder per capture day (`YYYY-MM-DD/`). Filenames encode the capture timestamp
(`YYYY-MM-DD_HHMM.jpg`). These are the source of truth from which the labeled YOLO
frames are derived, so the labeled dataset is reproducible from them.

Capture is performed on the Raspberry Pi by [`PYTHON/camera.py`](PYTHON/camera.py),
which takes a full-resolution still roughly every 10 minutes and writes it into the
current day's folder.

---

## Annotation format

- **`t10lot_labeled/crops_classifiers/{occupied,vacant}/`** - classification labels
  encoded by folder name (one crop per file).
- **`t10lot_labeled/crops_yolo_detect/labels/**/*.txt`** - YOLO labels, one file per
  image, one `class cx cy w h` row per annotated parking space.
- **`t10lot_labeled/vehicle_yolo_detect/labels/**/*.txt`** - YOLO labels, one file per
  image, one `0 cx cy w h` row per vehicle. An empty file is a frame with no vehicle
  in it and is kept on purpose.
