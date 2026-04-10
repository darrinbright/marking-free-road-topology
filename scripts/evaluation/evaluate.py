import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from scipy.spatial.distance import cdist
from tqdm import tqdm
from src.data.dataset import IDDTopoDataset
from src.data.graph import mask_to_graph, to_pyg
from src.models.htgnn import HTGNN

def collate(batch):
    return batch[0]

def seg_iou(pred_mask, gt_mask):
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)
    inter = (p & g).sum()
    union = (p | g).sum()
    if union == 0:
        return 1.0
    return float(inter / union)

def edge_precision_recall(pred_edges, gt_edges, pred_pos, gt_pos, thresh=0.15):
    if len(gt_edges) == 0:
        return (1.0, 1.0) if len(pred_edges) == 0 else (0.0, 0.0)
    if len(pred_edges) == 0:
        return (0.0, 0.0)
    pp = np.array(pred_pos)
    gp = np.array(gt_pos)

    def edge_midpoint(e, pos):
        return (pos[e[0]] + pos[e[1]]) / 2
    pred_mid = np.array([edge_midpoint(e, pp) for e in pred_edges])
    gt_mid = np.array([edge_midpoint(e, gp) for e in gt_edges])
    D = cdist(pred_mid, gt_mid)
    tp = 0
    used_gt = set()
    for i in range(len(pred_mid)):
        j = np.argmin(D[i])
        if D[i, j] < thresh and j not in used_gt:
            tp += 1
            used_gt.add(j)
    prec = tp / len(pred_edges) if len(pred_edges) > 0 else 0
    rec = tp / len(gt_edges) if len(gt_edges) > 0 else 0
    return (prec, rec)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ann_dir = ROOT / 'data' / 'annotations'
    idd_root = ROOT / 'data' / 'idd_raw'
    ckpt = ROOT / 'checkpoints' / 'best_gnn.pth'
    ds = IDDTopoDataset(str(ann_dir), str(idd_root))
    n = len(ds)
    _, _, test_ds = random_split(ds, [n - 45, 30, 15], generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate)
    model = HTGNN(backbone_ckpt=None).to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    ious = []
    edge_precs = []
    edge_recs = []
    with torch.no_grad():
        for b in tqdm(test_loader, desc='Evaluating'):
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            model.memory.reset()
            preds = model(b)
            pred_seg = F.interpolate(preds['seg_logits'], size=b['bev_mask'].shape[1:], mode='bilinear', align_corners=False).argmax(dim=1).float()
            gt = b['bev_mask'].squeeze(0).cpu().numpy()
            pred_np = pred_seg.squeeze(0).cpu().numpy().astype(np.uint8)
            iou = seg_iou(pred_np, gt)
            ious.append(iou)
            pred_mask_bev = pred_np.astype(np.uint8)
            try:
                G_pred = mask_to_graph(pred_mask_bev)
                pred_node_feats, pred_edge_index = to_pyg(G_pred)
                pred_node_feats = pred_node_feats.numpy()
                pred_edges = list(zip(pred_edge_index[0].tolist(), pred_edge_index[1].tolist()))
                pred_pos = pred_node_feats[:, :2].tolist()
            except Exception:
                pred_edges = []
                pred_pos = []
            node_feats = b['node_feats'].cpu().numpy()
            edge_index = b['edge_index'].cpu().numpy()
            gt_edges = list(zip(edge_index[0], edge_index[1]))
            gt_pos = node_feats[:, :2].tolist()
            prec, rec = edge_precision_recall(pred_edges, gt_edges, pred_pos, gt_pos)
            edge_precs.append(prec)
            edge_recs.append(rec)
    n_test = len(ious)
    mean_iou = np.mean(ious)
    mean_eprec = np.mean(edge_precs)
    mean_erec = np.mean(edge_recs)
    edge_f1 = 2 * mean_eprec * mean_erec / (mean_eprec + mean_erec) if mean_eprec + mean_erec > 0 else 0.0
    results = {'iou': float(mean_iou), 'edge_precision': float(mean_eprec), 'edge_recall': float(mean_erec), 'edge_f1': float(edge_f1), 'n_test': n_test}
    import json
    out = ROOT / 'outputs' / 'results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    return results
if __name__ == '__main__':
    main()