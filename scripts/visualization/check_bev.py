import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.data.bev import to_bev, mask_to_bev
from src.data.dataset import IDDTopoDataset

def _rasterize_mask(ann_path, h, w):
    with open(ann_path) as f:
        ann = json.load(f)
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in ann.get('shapes', []):
        if shape.get('label') != 'passable_surface':
            continue
        pts = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask

def main():
    ann_dir = ROOT / 'data' / 'annotations'
    idd_root = ROOT / 'data' / 'idd_raw'
    ds = IDDTopoDataset(str(ann_dir), str(idd_root))
    if len(ds) == 0:
        return
    indices = [0, len(ds) // 5, 2 * len(ds) // 5, 3 * len(ds) // 5, 4 * len(ds) // 5, len(ds) - 1]
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    for col, idx in enumerate(indices):
        ann_path, img_path = ds.samples[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = _rasterize_mask(ann_path, h, w)
        bev_img = to_bev(img)
        bev_mask = mask_to_bev(mask, h, w)
        axes[0, col].imshow(img)
        axes[0, col].set_title('Original')
        axes[0, col].axis('off')
        axes[1, col].imshow(bev_img)
        axes[1, col].set_title('BEV')
        axes[1, col].axis('off')
        overlay = bev_img.copy()
        overlay[bev_mask > 0] = overlay[bev_mask > 0] * 0.5 + np.array([0, 200, 0])
        axes[2, col].imshow(overlay)
        axes[2, col].set_title('BEV + mask')
        axes[2, col].axis('off')
    plt.tight_layout()
    out = ROOT / 'bev_calibration_check.png'
    plt.savefig(out, dpi=100)
if __name__ == '__main__':
    main()