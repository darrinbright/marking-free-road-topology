import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from src.data.dataset import IDDTopoDataset
from src.data.bev import to_bev
from src.data.graph import mask_to_graph, to_pyg
from src.models.htgnn import HTGNN

def collate(batch):
    return batch[0]

def draw_graph_on_bev(bev_img, node_feats, edge_index, color=(0, 255, 0), node_color=(255, 0, 0)):
    H, W = bev_img.shape[:2]
    if isinstance(bev_img, np.ndarray) and bev_img.ndim == 2:
        overlay = cv2.cvtColor(bev_img, cv2.COLOR_GRAY2BGR)
    else:
        overlay = bev_img.copy() if bev_img.ndim == 3 else cv2.cvtColor(bev_img, cv2.COLOR_GRAY2BGR)
    pos = node_feats[:, :2].numpy()
    pts = (pos * np.array([W - 1, H - 1])).astype(np.int32)
    for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        pt1 = (int(pts[i, 0]), int(pts[i, 1]))
        pt2 = (int(pts[j, 0]), int(pts[j, 1]))
        cv2.line(overlay, pt1, pt2, color, 2)
    for i, pt in enumerate(pts):
        c = (255, 100, 100) if node_feats[i, 3] > 0.5 else node_color
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 4, c, -1)
    return overlay

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ann_dir = ROOT / 'data' / 'annotations'
    idd_root = ROOT / 'data' / 'idd_raw'
    ckpt = ROOT / 'checkpoints' / 'best_gnn.pth'
    backbone_ckpt = ROOT / 'checkpoints' / 'best_backbone.pth'
    ds = IDDTopoDataset(str(ann_dir), str(idd_root))
    n = len(ds)
    straight, curves, junctions = ([], [], [])
    for idx in range(n):
        sample = ds[idx]
        node_feats = sample['node_feats'].numpy()
        n_nodes = len(node_feats)
        has_junc = (node_feats[:, 3] > 0.5).any()
        if has_junc:
            junctions.append(idx)
        elif n_nodes <= 4:
            straight.append(idx)
        else:
            curves.append(idx)
    rng = np.random.RandomState(42)

    def pick(pool, k=2):
        if len(pool) >= k:
            return list(rng.choice(pool, k, replace=False))
        return list(pool)[:k]
    straight_idx = pick(straight)
    curve_idx = pick(curves)
    junction_idx = pick(junctions)
    all_picked = straight_idx + curve_idx + junction_idx
    while len(all_picked) < 6:
        rest = [i for i in range(n) if i not in all_picked]
        if not rest:
            break
        all_picked.append(rng.choice(rest))
    indices = (straight_idx + curve_idx + junction_idx)[:6]
    labels = ['Straight'] * len(straight_idx) + ['Curve'] * len(curve_idx) + ['Junction'] * len(junction_idx)
    labels = (labels + ['Other'] * 6)[:6]
    model = HTGNN(backbone_ckpt=str(backbone_ckpt) if backbone_ckpt.exists() else None).to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    out_dir = ROOT / 'outputs' / 'qualitative'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle('HT-GNN Qualitative Results: Original → BEV → Predicted Graph Overlay', fontsize=12)
    with torch.no_grad():
        for col, (idx, label) in enumerate(zip(indices, labels)):
            sample = ds[idx]
            ann_path, img_path = ds.samples[idx]
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            bev_img = to_bev(img)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            model.memory.reset()
            preds = model(batch)
            pred_seg = F.interpolate(preds['seg_logits'], size=(256, 256), mode='bilinear', align_corners=False).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            G = mask_to_graph(pred_seg)
            pred_node_feats, pred_edge_index = to_pyg(G)
            overlay = draw_graph_on_bev(bev_img, pred_node_feats, pred_edge_index, color=(0, 255, 0))
            axes[0, col].imshow(img)
            axes[0, col].set_title(f'{label} (orig)')
            axes[0, col].axis('off')
            axes[1, col].imshow(bev_img)
            axes[1, col].set_title('BEV')
            axes[1, col].axis('off')
            axes[2, col].imshow(overlay)
            axes[2, col].set_title('Pred graph')
            axes[2, col].axis('off')
    plt.tight_layout()
    out_path = out_dir / 'qualitative_6.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    for col, (idx, label) in enumerate(zip(indices, labels)):
        sample = ds[idx]
        ann_path, img_path = ds.samples[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        bev_img = to_bev(img)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        model.memory.reset()
        preds = model(batch)
        pred_seg = F.interpolate(preds['seg_logits'], size=(256, 256), mode='bilinear', align_corners=False).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        G = mask_to_graph(pred_seg)
        pred_node_feats, pred_edge_index = to_pyg(G)
        overlay = draw_graph_on_bev(bev_img, pred_node_feats, pred_edge_index, color=(0, 255, 0))
        gt_node_feats = sample['node_feats']
        gt_edge_index = sample['edge_index']
        gt_overlay = draw_graph_on_bev(bev_img, gt_node_feats, gt_edge_index, color=(0, 0, 255), node_color=(0, 255, 0))
        fig2, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img)
        ax[0].set_title('Original RGB')
        ax[0].axis('off')
        ax[1].imshow(gt_overlay)
        ax[1].set_title('Ground Truth Graph')
        ax[1].axis('off')
        ax[2].imshow(overlay)
        ax[2].set_title('Predicted Graph (Ours)')
        ax[2].axis('off')
        stem = Path(ann_path).stem
        fig2.suptitle(f'{label}: {stem}')
        plt.tight_layout()
        fig2.savefig(out_dir / f'fig_{col + 1}_{label.lower()}_{stem}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
if __name__ == '__main__':
    main()