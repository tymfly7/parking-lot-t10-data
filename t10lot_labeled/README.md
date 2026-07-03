# t10lot_labeled — labeled datasets

This folder holds the labeled data for the **Parking-Lot-T10** dataset captured at
parking lot T10, **Brno University of Technology (BUT)**.

- `crops_classifiers/` — classification dataset, per-slot crops sorted into
  `occupied/` (61,057) and `vacant/` (36,152) folders — 97,209 total.
- `crops_yolo_detect/` — detection dataset in Ultralytics **YOLO** format
  (`vacant` / `occupied` boxes), 3,137 frames split 2195 train / 470 val / 472 test.

Full documentation — source & attribution, directory layout, and training commands —
is in the repository's main [README](../README.md).
