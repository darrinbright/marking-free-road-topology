import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import random_split
from src.data.dataset import IDDTopoDataset

def main():
    ann_dir = ROOT / 'data' / 'annotations'
    idd_root = ROOT / 'data' / 'idd_raw'
    ds = IDDTopoDataset(str(ann_dir), str(idd_root))
    n = len(ds)
    train_ds, val_ds, test_ds = random_split(ds, [n - 45, 30, 15], generator=torch.Generator().manual_seed(42))

    def count_junctions(subset):
        has_junc = 0
        total = len(subset)
        for idx in subset.indices:
            sample = ds[idx]
            node_feats = sample['node_feats']
            junctions = (node_feats[:, 3] > 0.5).any().item()
            if junctions:
                has_junc += 1
        return (has_junc, total)
    t_j, t_n = count_junctions(train_ds)
    v_j, v_n = count_junctions(val_ds)
    te_j, te_n = count_junctions(test_ds)
    total_junc = t_j + v_j + te_j
if __name__ == '__main__':
    main()