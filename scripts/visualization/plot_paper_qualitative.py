import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT = Path('/home/darrin/Desktop/Automotive2')
sys.path.insert(0, str(ROOT))

from src.data.dataset import IDDTopoDataset
from src.data.bev import to_bev
from src.data.graph import mask_to_graph, to_pyg
from src.models.htgnn import HTGNN
from scripts.visualization.visualize_qualitative import draw_graph_on_bev

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ann_dir = ROOT / 'data' / 'annotations'
    idd_root = ROOT / 'data' / 'idd_raw'
    ckpt = ROOT / 'checkpoints' / 'best_gnn_temporal.pth'
    backbone_ckpt = ROOT / 'checkpoints' / 'best_backbone.pth'

    ds = IDDTopoDataset(str(ann_dir), str(idd_root))
    model = HTGNN(backbone_ckpt=str(backbone_ckpt) if backbone_ckpt.exists() else None, use_temporal=True).to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    target_stems = ["132_867435", "132_944936", "132_483549"]
    target_indices = []
    for i in range(len(ds)):
        stem = Path(ds.samples[i][0]).stem
        if stem in target_stems:
            target_indices.append(i)
            
    fig, axes = plt.subplots(3, 5, figsize=(25, 12))
    
    for row, idx in enumerate(target_indices[:3]):
        sample = ds[idx]
        ann_path, img_path = ds.samples[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        bev_img = to_bev(img)
        
        gt_node_feats = sample["node_feats"]
        gt_edge_index = sample["edge_index"]
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        model.memory.reset()
        
        with torch.no_grad():
            preds = model(batch)
            pred_seg = F.interpolate(preds['seg_logits'], size=(256, 256), mode='bilinear', align_corners=False).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            from skimage.morphology import skeletonize, closing, disk
            clean = closing(pred_seg.astype(bool), footprint=disk(5))
            skel = skeletonize(clean).astype(np.uint8)
            
            G = mask_to_graph(pred_seg)
            pred_node_feats, pred_edge_index = to_pyg(G)
            
            mask_overlay = bev_img.copy()
            mask_overlay[pred_seg == 1] = mask_overlay[pred_seg == 1] * 0.5 + np.array([0, 0, 255]) * 0.5
            
            skel_overlay = bev_img.copy()
            skel_overlay[skel == 1] = [0, 255, 0]
            
            overlay = draw_graph_on_bev(bev_img, pred_node_feats, pred_edge_index, color=(0, 255, 0))
            gt_overlay = draw_graph_on_bev(bev_img, gt_node_feats, gt_edge_index, color=(0, 0, 255), node_color=(0, 255, 0))
            
            axes[row, 0].imshow(img)
            axes[row, 1].imshow(mask_overlay.astype(np.uint8))
            axes[row, 2].imshow(skel_overlay)
            axes[row, 3].imshow(gt_overlay)
            axes[row, 4].imshow(overlay)
            
            for col in range(5):
                axes[row, col].axis("off")
                
            if row == 0:
                axes[row, 0].set_title("Original Image", fontsize=20, fontweight='bold', pad=15)
                axes[row, 1].set_title("Polygon Segmentation", fontsize=20, fontweight='bold', pad=15)
                axes[row, 2].set_title("Skeleton Extraction", fontsize=20, fontweight='bold', pad=15)
                axes[row, 3].set_title("Ground Truth Topology", fontsize=20, fontweight='bold', pad=15)
                axes[row, 4].set_title("Prediction (Ours)", fontsize=20, fontweight='bold', pad=15)

    plt.tight_layout(pad=1.0)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "visualization_figures.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved qualitative paper figure to {out_path}")

if __name__ == '__main__':
    main()
