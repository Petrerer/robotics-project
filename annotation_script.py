import os
import glob
import random
import shutil
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Output directory expected by XYDataset
OUTPUT_DIR = 'datasets/dataset_labeled_5'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'labels.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset_root = 'dataset'
session_csvs = glob.glob(os.path.join(dataset_root, '*.csv'))
copied, skipped = 0, 0

# Load existing labels to allow resuming
if os.path.exists(OUTPUT_CSV):
    existing_df = pd.read_csv(OUTPUT_CSV)
    labeled_files = set(existing_df['image_path'].tolist())
    all_rows = existing_df.to_dict('records')
    print(f"Resuming — {len(labeled_files)} images already labeled.")
else:
    labeled_files = set()
    all_rows = []

def save_labels():
    df = pd.DataFrame(all_rows, columns=['image_path', 'x_norm', 'y_norm'])
    df.to_csv(OUTPUT_CSV, index=False)

def get_click(img_array, title, index, total):
    result = {'coords': None, 'action': None}

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_array)
    ax.set_title(f"[{index}/{total}] {title}  |  Click=label  •  S=skip  •  Q=quit", fontsize=10)
    ax.axis('on')

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        x_px, y_px = event.xdata, event.ydata
        h, w = img_array.shape[:2]
        result['coords'] = (x_px / w, y_px / h)
        plt.close(fig)   # single click → done immediately

    def on_key(event):
        if event.key == 's':
            result['action'] = 'skip'
            plt.close(fig)
        elif event.key == 'q':
            result['action'] = 'quit'
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()   # blocks until window is closed

    if result['action']:
        return result['action']
    if result['coords'] is None:
        return 'skip'   # closed without clicking or key
    return result['coords']


# ── Build the full work list, then shuffle it ──────────────────────────────
work_items = []   # list of (src_path, dest_path, session_name, frame_id)

for csv_path in session_csvs:
    session_name = os.path.splitext(os.path.basename(csv_path))[0]
    image_dir = os.path.join(dataset_root, session_name)

    if not os.path.isdir(image_dir):
        print(f"Skipping {session_name} — no matching image folder found")
        continue

    df = pd.read_csv(csv_path, header=None, names=['frame', 'x', 'y'])
    for _, row in df.iterrows():
        frame_id = int(row['frame'])
        src = os.path.join(image_dir, f'{frame_id:04d}.jpg')
        dest_name = f'{session_name}_{frame_id:04d}.jpg'
        dest_path = os.path.join(OUTPUT_DIR, dest_name)

        if not os.path.exists(src):
            skipped += 1
            continue

        # Skip already-labeled images before we even show them
        if dest_path in labeled_files:
            copied += 1
            continue

        work_items.append((src, dest_path, session_name, frame_id))

random.shuffle(work_items)
total = len(work_items)
print(f"{total} unlabeled images to annotate.")

# ── Main annotation loop ───────────────────────────────────────────────────
for idx, (src, dest_path, session_name, frame_id) in enumerate(work_items, start=1):
    img = Image.open(src).convert('RGB')
    img_array = np.array(img)

    title = f"[{session_name}] frame {frame_id:04d}"
    outcome = get_click(img_array, title, idx, total)

    if outcome == 'quit':
        print("\nLabeling interrupted by user. Progress saved.")
        save_labels()
        print(f"Done: {copied} images copied, {skipped} skipped, "
              f"{len(all_rows)} labeled so far.")
        exit(0)

    if outcome == 'skip':
        print(f"  Skipped frame {frame_id:04d}")
        skipped += 1
        continue

    x_norm, y_norm = outcome

    # Copy image to output directory
    shutil.copy2(src, dest_path)

    # Record label
    all_rows.append({
        'image_path': dest_path,
        'x_norm': round(x_norm, 6),
        'y_norm': round(y_norm, 6),
    })
    labeled_files.add(dest_path)
    copied += 1

    # Save after every annotation so progress is never lost
    save_labels()
    print(f"  [{idx}/{total}] Labeled {session_name}/{frame_id:04d} → ({x_norm:.4f}, {y_norm:.4f})")

print(f"\nDone: {copied} images labeled, {skipped} skipped.")
print(f"Labels saved to: {OUTPUT_CSV}")
