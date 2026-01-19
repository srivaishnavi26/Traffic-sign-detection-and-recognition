# Real-Time Traffic Sign Recognition with Open-Set Rejection

## Overview

This project implements a real-time Traffic Sign Recognition (TSR)
pipeline for roadside videos. The system combines object detection,
fine-grained classification, open-set rejection, and temporal consensus
to achieve stable and reliable predictions in real-world driving
scenarios.

The focus is on robustness under challenging conditions such as small
objects, motion blur, partial occlusion, and visually similar
non-traffic signage.

------------------------------------------------------------------------

## Pipeline Architecture

1.  **Traffic Sign Detection**
    -   YOLO-based detector optimized for small traffic signs
    -   Generates bounding boxes for candidate signs in each frame
2.  **Traffic Sign Classification**
    -   ResNet18 classifier trained on GTSRB-style cropped signs
    -   Supports 43 traffic sign classes
3.  **Open-Set Rejection**
    -   Confidence-based rejection using softmax probabilities
    -   Low-confidence predictions are labeled as `UNKNOWN`
4.  **Temporal Consensus**
    -   Sliding-window label smoothing across frames
    -   IoU-based tracking to maintain stable identities per sign

------------------------------------------------------------------------

## Dataset

-   **Primary Dataset:** German Traffic Sign Recognition Benchmark
    (GTSRB)
-   **Additional Data:** Cropped detections from real roadside videos
-   Data organized using ImageFolder structure for training and
    validation

------------------------------------------------------------------------

## Evaluation

### Classification Performance

-   Validation Accuracy: \~100% on curated cropped signs
-   Confusion Matrix: `runs/eval/confusion_matrix.png`

### Open-Set Evaluation

-   Open-set AUROC: **1.00**
-   ROC Curve: `runs/eval/open_set_roc.png`

> Note: Near-perfect results reflect the controlled nature of cropped
> validation data. Open-set rejection and temporal consensus are
> critical for real-world deployment.

------------------------------------------------------------------------

## Results

-   Stable traffic sign labels across video frames
-   Reduced false positives through confidence-based rejection
-   Robust performance under motion blur and partial occlusion

------------------------------------------------------------------------

## How to Run

``` bash
# Run detection (example)
yolo detect predict model=your_model.pt source=data/raw/road_videos

# Run full video pipeline
python src/postprocess/overlay_all_videos_bbox.py

# Evaluation
python src/eval/eval_classifier_on_crops.py
python src/eval/eval_open_set_auroc.py
```

------------------------------------------------------------------------

## Project Structure

    src/
     ├── classifier/
     ├── postprocess/
     │    ├── temporal_consensus.py
     │    ├── iou_tracker.py
     │    └── overlay_all_videos_bbox.py
     └── eval/
          ├── eval_classifier_on_crops.py
          └── eval_open_set_auroc.py

------------------------------------------------------------------------

## Applications

-   Driver Assistance Systems (ADAS)
-   Intelligent Transportation Systems (ITS)
-   Open-world traffic sign recognition research

------------------------------------------------------------------------

## Future Work

-   Hard-negative mining with non-traffic signage
-   Deployment on embedded devices
-   Integration with map-based navigation systems
