# Robotics II: Deep Learning-Based Autonomous Road-Following for Jetbot

This repository contains the dataset annotation, model training, benchmarking, and real-time execution pipeline for a Jetbot robot designed to autonomously follow a track.

For a comprehensive analysis of the system architecture, datasets, GPU benchmarks, control theory, and recommendations, please refer to the detailed [Project Report (report.md)](file:///Users/daniel/Developer/Projects/robotics-project/report.md).

---

## 🚀 Quick Start & Run Instructions

All local preprocessing, labeling, and benchmarking scripts are organized and fully runnable using the `uv` package manager.

### 1. Project Setup
Ensure you have `uv` installed, then run the commands from the root directory.

### 2. Interactive Annotation (Dataset Labeling)
If you want to annotate new frames or review the labeling interface, run:
```bash
uv run python3 annotation_script.py
```
* **Controls**: Click on the image to place a target steering coordinate, press `S` to skip, or press `Q` to save and quit. Labels are written to `datasets/dataset_labeled_5/labels.csv`.

### 3. Model Training
Open the training notebook [train_model.ipynb](file:///Users/daniel/Developer/Projects/robotics-project/train_model.ipynb) in your Jupyter environment.
* Configured for **ResNet-18** and **MobileNet-V2** architectures with input scaling, color jitter, cropping, and horizontal flips.
* Checkpoints are stored in the `models/` directory.

### 4. Local Testing & Benchmarking
Open the test notebook [model_testing.ipynb](file:///Users/daniel/Developer/Projects/robotics-project/model_testing.ipynb) to:
* Benchmark CPU/GPU inference latency and throughput (FPS).
* Visualize predicted steering target coordinates overlaid on sample images.

### 5. Running on the Jetbot Track
1. Upload [live_demo.ipynb](file:///Users/daniel/Developer/Projects/robotics-project/live_demo.ipynb) and your chosen `.pth` model (from the `models/` folder) to the Jetbot.
2. Update the path to the model in the notebook.
3. Adjust the control parameters:
   * `SPEED_GAIN` (motor speed limit)
   * `STEERING_GAIN` (proportional steering coefficient)
   * `STEERING_DGAIN` (derivative coefficient to damp oscillations)
   * `STEERING_BIAS` (to align drift)
4. Execute the cells to activate camera-to-motor execution loop.

---

## 📊 Summary of Datasets & Models

* **Datasets**: Located in [datasets/](file:///Users/daniel/Developer/Projects/robotics-project/datasets/). Includes datasets targeting points closer to the robot for high stability (`dataset_labeled_1`) and datasets targeting far look-ahead points for fast corner-cutting (`dataset_labeled_2`).
* **Pretrained Models**: Located in [models/](file:///Users/daniel/Developer/Projects/robotics-project/models/). Contains ready-to-run models for ResNet-18 and MobileNet-V2 at various resolutions.

For full results and mathematical details, read [report.md](file:///Users/daniel/Developer/Projects/robotics-project/report.md).
