# Robotics II — JetBot Road Following

Autonomous track-following for NVIDIA JetBot using a ResNet-18 model trained on manually annotated camera frames.

**Team:** Piotr Franc, Jakub Adamski, Piotr Foltyniewicz, Wiktor Talarek, Daniel Skalski

---

## Requirements

- Python 3 with `uv` installed (local machine)
- JetBot with CUDA-capable GPU (required for on-robot inference)
- PyTorch, torchvision, PIL, pandas, matplotlib, numpy

---

## 1. Annotate Images

Place your session folders and CSVs under `dataset/`, then run:

```bash
uv run python3 annotation_script.py
```

- **Click** on the image to place a steering target point
- **S** — skip the current frame
- **Q** — save progress and quit

Labels are saved to `datasets/dataset_labeled_5/labels.csv`. The script resumes where it left off if you run it again.

---

## 2. Train the Model

Open `train_model.ipynb` and set `DATASET_NAME` to your dataset folder, then run all cells.

- Architecture: ResNet-18 pretrained on ImageNet, final layer replaced with `Linear(512, 2)`
- Optimizer: Adam, 70 epochs, batch size 8, MSE loss
- Best checkpoint saved to `models/<DATASET_NAME>/best_steering_model_xy.pth`

After training, the notebook also exports a legacy-format model for JetBot compatibility:

```python
torch.save(state_dict, 'model_to_import.pth', _use_new_zipfile_serialization=False)
```

Use `model_to_import.pth` when uploading to the robot.

---

## 3. Test Locally

Open `model_testing.ipynb` to run inference on CPU without the robot. Set `MODEL_PATH` and `DATASET_NAME` at the top, then run all cells to visualise predicted steering points and benchmark inference speed.

---

## 4. Run on the JetBot

1. Upload `live_demo_executelock.ipynb` and `model_to_import.pth` to the JetBot
2. Set `MODEL_PATH` in the notebook to point to your uploaded `.pth` file
3. Run all cells — the robot will start following the track

**Tunable parameters** (adjust until the robot runs smoothly):

| Parameter | Description | Starting value |
|---|---|---|
| `SPEED_GAIN` | Base motor speed | ~0.28 |
| `STEERING_GAIN` | Proportional steering strength | ~0.09 |
| `STEERING_DGAIN` | Derivative term to reduce oscillation | 0.0 |
| `STEERING_BIAS` | Offset to correct left/right drift | 0.0 |

> **Important:** Always stop the notebook by running the stop cell before closing the browser tab. Closing the tab without stopping will freeze the camera and require a full JetBot reboot.

---

## Project Structure

```
├── annotation_script.py       # Manual labelling tool
├── train_model.ipynb          # Model training
├── model_testing.ipynb        # Local inference & benchmarking
├── live_demo_executelock.ipynb # On-robot execution loop
├── dataset/                   # Raw session frames and CSVs from PUT
├── datasets/                  # Annotated datasets (labels.csv + images)
└── models/                    # Saved model checkpoints
```