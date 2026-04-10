import sys
from pathlib import Path
ROOT = Path('/home/darrin/Desktop/Automotive2')
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.data.dataset import IDDTopoDataset
from src.data.bev import to_bev
from src.data.graph import mask_to_graph, to_pyg
from src.models.htgnn import HTGNN
from scripts.visualize_qualitative import draw_graph_on_bev

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
    out_dir = ROOT / 'outputs' / 'qualitative_best'
    out_dir.mkdir(parents=True, exist_ok=True)
    best_count = 0
    rng = np.random.RandomState(42)
    indices = np.arange(len(ds))
    rng.shuffle(indices)
    for idx in indices:
        sample = ds[idx]
        gt_edge_index = sample['edge_index']
        if gt_edge_index.shape[1] < 15:
            continue
        gt_node_feats = sample['node_feats']
        if not (gt_node_feats[:, 3] > 0.5).any():
            continue
        ann_path, img_path = ds.samples[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        bev_img = to_bev(img)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        model.memory.reset()
        with torch.no_grad():
            preds = model(batch)
            pred_seg = F.interpolate(preds['seg_logits'], size=(256, 256), mode='bilinear', align_corners=False).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            G = mask_to_graph(pred_seg)
            pred_node_feats, pred_edge_index = to_pyg(G)
            if pred_edge_index.shape[1] < 10:
                continue
            overlay = draw_graph_on_bev(bev_img, pred_node_feats, pred_edge_index, color=(0, 255, 0))
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
            plt.tight_layout()
            out_file = out_dir / f'perfect_{best_count + 1}_{stem}.png'
            fig2.savefig(out_file, dpi=150, bbox_inches='tight')
            plt.close()
            best_count += 1
            if best_count >= 6:
                break
if __name__ == '__main__':
    main()