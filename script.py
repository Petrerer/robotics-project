import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np
import pandas as pd
import random
from datetime import datetime

DATASET_NAME = 'dataset_labeled_2'
DATASET_DIR = 'datasets/' + DATASET_NAME

_label_cache = {}
def _load_labels(csv_file):
    if csv_file not in _label_cache:
        df = pd.read_csv(csv_file)
        _label_cache[csv_file] = {os.path.basename(r['image_path']): (r['x_norm'], r['y_norm']) for _, r in df.iterrows()}
    return _label_cache[csv_file]

def get_x(path, width, csv_file=f'{DATASET_DIR}/labels.csv'):
    return _load_labels(csv_file)[os.path.basename(path)][0]

def get_y(path, height, csv_file=f'{DATASET_DIR}/labels.csv'):
    return _load_labels(csv_file)[os.path.basename(path)][1]

def augment_image(image, x, y, max_crop=0.8):
    W, H = image.size
    scale = np.random.uniform(max_crop, 1.0)
    crop_w, crop_h = int(W * scale), int(H * scale)
    left = np.random.randint(0, W - crop_w + 1)
    top  = np.random.randint(0, H - crop_h + 1)
    image = transforms.functional.crop(image, top, left, crop_h, crop_w)
    x_new = (x * W - left) / crop_w
    y_new = (y * H - top)  / crop_h
    image = transforms.functional.resize(image, (H, W))
    return image, x_new, y_new

class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, resolution, random_hflips=False, augment=False):
        self.directory = directory
        self.resolution = resolution
        self.random_hflips = random_hflips
        self.augment = augment
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = PIL.Image.open(image_path)
        image = transforms.functional.resize(image, (self.resolution, self.resolution))
        width, height = image.size
        x = float(get_x(os.path.basename(image_path), width))
        y = float(get_y(os.path.basename(image_path), height))
        if self.random_hflips and float(np.random.rand(1)) > 0.5:
            image = transforms.functional.hflip(image)
            x = 1 - x
        if self.augment:
            image, x, y = augment_image(image, x, y)
        image = self.color_jitter(image)
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.tensor([x, y]).float()

os.makedirs(f'models/{DATASET_NAME}', exist_ok=True)
device = torch.device('cpu')
results = []

for resolution in [224, 128, 64]:
    print(f'\n{"="*50}\nResolution: {resolution}x{resolution}\n{"="*50}')

    dataset = XYDataset(DATASET_DIR, resolution=resolution, random_hflips=True, augment=True)
    num_train = int(0.8 * len(dataset))
    num_val   = int(0.1 * len(dataset))
    num_test  = len(dataset) - num_train - num_val
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=0)

    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 2)
    model = model.to(device)

    BEST_MODEL_PATH = f'models/{DATASET_NAME}/best_model_{resolution}.pth'
    NUM_EPOCHS = 70
    best_loss = 1e9
    optimizer = optim.Adam(model.parameters())

    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                val_loss += float(F.mse_loss(model(images.to(device)), labels.to(device)))
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_loss = val_loss

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/{NUM_EPOCHS} — Val Loss: {val_loss:.4f}')

    # --- Evaluate on test set ---
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))
    model.eval()

    def get_image_tensor(image_path, res):
        transform = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(PIL.Image.open(image_path).convert('RGB'))

    samples = [dataset.image_paths[i] for i in test_loader.dataset.indices]
    random.shuffle(samples)

    euclidean_errors = []
    inference_times  = []

    for s in samples:
        image = PIL.Image.open(s)
        width, height = image.size
        true_x = get_x(os.path.basename(s), width)
        true_y = get_y(os.path.basename(s), height)

        t0 = datetime.now()
        with torch.no_grad():
            output = model(get_image_tensor(s, resolution).unsqueeze(0))[0].numpy()
        inference_times.append((datetime.now() - t0).total_seconds())

        pred_x, pred_y = output[0], output[1]
        # Euclidean distance in normalised [0,1]^2 space, rescaled to pixels at 224
        dist_norm = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        euclidean_errors.append(dist_norm * 224)   # "pixels at 224px reference"

    avg_time  = np.mean(inference_times)
    avg_dist  = np.mean(euclidean_errors)
    fps       = 1.0 / avg_time

    print(f'  Avg inference time : {avg_time*1000:.1f} ms  ({fps:.1f} FPS)')
    print(f'  Avg Euclidean error: {avg_dist:.2f} px (224px ref)')

    results.append({
        'Resolution': f'{resolution}×{resolution}',
        'Avg Inference (ms)': round(avg_time * 1000, 1),
        'FPS': round(fps, 1),
        'Avg Euclidean Error (px@224)': round(avg_dist, 2),
    })

# --- Summary table ---
print('\n\n' + '='*60)
print('RESOLUTION BENCHMARK SUMMARY')
print('='*60)
df = pd.DataFrame(results)
print(df.to_string(index=False))
print('='*60)
print('* Euclidean error is the average predicted-vs-true distance,')
print('  expressed in pixels relative to a 224×224 reference frame.')