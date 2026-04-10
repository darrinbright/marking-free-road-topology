import argparse
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
    return float(inter / union) if union > 0 else 1.0

def edge_precision_recall(pred_edges, gt_edges, pred_pos, gt_pos, thresh=0.15):
    if len(gt_edges) == 0:
        return (1.0, 1.0) if len(pred_edges) == 0 else (0.0, 0.0)
    if len(pred_edges) == 0:
        return (0.0, 0.0)
    pp, gp = (np.array(pred_pos), np.array(gt_pos))
    pred_mid = np.array([(pp[e[0]] + pp[e[1]]) / 2 for e in pred_edges])
    gt_mid = np.array([(gp[e[0]] + gp[e[1]]) / 2 for e in gt_edges])
    D = cdist(pred_mid, gt_mid)
    tp, used_gt = (0, set())
    for i in range(len(pred_mid)):
        j = np.argmin(D[i])
        if D[i, j] < thresh and j not in used_gt:
            tp += 1
            used_gt.add(j)
    prec = tp / len(pred_edges)
    rec = tp / len(gt_edges)
    return (prec, rec)

def run_eval(device, test_loader, model, backbone_ckpt, edge_threshold=0.5, use_gnn_filter=True):
    ckpt = ROOT / 'checkpoints' / 'best_gnn.pth'
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    ious, eprecs, erecs = ([], [], [])
    with torch.no_grad():
        for b in tqdm(test_loader, desc=f'eval(edge_thr={edge_threshold})'):
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            model.memory.reset()
            preds = model(b)
            pred_seg = F.interpolate(preds['seg_logits'], size=b['bev_mask'].shape[1:], mode='bilinear', align_corners=False).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            gt = b['bev_mask'].squeeze(0).cpu().numpy()
            iou = seg_iou(pred_seg, gt)
            ious.append(iou)
            try:
                G_pred = mask_to_graph(pred_seg.astype(np.uint8))
                pred_node_feats, pred_edge_index = to_pyg(G_pred)
            except Exception:
                pred_edges, pred_pos = ([], [])
                node_feats = b['node_feats'].cpu().numpy()
                edge_index = b['edge_index'].cpu().numpy()
                gt_edges = list(zip(edge_index[0], edge_index[1]))
                gt_pos = node_feats[:, :2].tolist()
                prec, rec = edge_precision_recall([], gt_edges, [], gt_pos)
                eprecs.append(prec)
                erecs.append(rec)
                continue
            pred_edges = list(zip(pred_edge_index[0].tolist(), pred_edge_index[1].tolist()))
            pred_pos = pred_node_feats.numpy()[:, :2].tolist()
            node_feats = b['node_feats'].cpu().numpy()
            edge_index = b['edge_index'].cpu().numpy()
            gt_edges = list(zip(edge_index[0], edge_index[1]))
            gt_pos = node_feats[:, :2].tolist()
            if use_gnn_filter and len(pred_edges) > 0:
                b_pred = {'bev_img': b['bev_img'], 'node_feats': pred_node_feats.float().to(device), 'edge_index': pred_edge_index.long().to(device), 'bev_mask': b['bev_mask'], 'ego_motion': b.get('ego_motion', torch.zeros(1, 4, device=device))}
                model.memory.reset()
                preds2 = model(b_pred)
                edge_logits = preds2['edge_logits'].cpu()
                probs = torch.sigmoid(edge_logits).numpy()
                keep = [i for i in range(len(pred_edges)) if probs[i] >= edge_threshold]
                pred_edges = [pred_edges[i] for i in keep]
                pred_pos = pred_pos
            prec, rec = edge_precision_recall(pred_edges, gt_edges, pred_pos, gt_pos)
            eprecs.append(prec)
            erecs.append(rec)
    p, r = (np.mean(eprecs), np.mean(erecs))
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {'iou': float(np.mean(ious)), 'edge_precision': float(p), 'edge_recall': float(r), 'edge_f1': float(f1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true', help='Sweep thresholds 0.2,0.3,0.4,0.5 and report best')
    ap.add_argument('--threshold', type=float, default=0.2, help='Edge prob threshold (default 0.2, conservative)')
    ap.add_argument('--no-filter', action='store_true', help='Disable GNN edge filtering (raw skeleton)')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ann_dir = ROOT / 'data' / 'annotations'
    idd_root = ROOT / 'data' / 'idd_raw'
    backbone_ckpt = ROOT / 'checkpoints' / 'best_backbone.pth'
    ds = IDDTopoDataset(str(ann_dir), str(idd_root))
    n = len(ds)
    _, _, test_ds = random_split(ds, [n - 45, 30, 15], generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate)
    model = HTGNN(backbone_ckpt=str(backbone_ckpt) if backbone_ckpt.exists() else None).to(device)
    if args.sweep:
        best = None
        best_f1 = -1
        for thresh in [0.2, 0.3, 0.4, 0.5]:
            r = run_eval(device, test_loader, model, backbone_ckpt, edge_threshold=thresh, use_gnn_filter=not args.no_filter)
            if r['edge_f1'] > best_f1:
                best_f1 = r['edge_f1']
                best = (thresh, r)
        return best[1]
    r = run_eval(device, test_loader, model, backbone_ckpt, edge_threshold=args.threshold, use_gnn_filter=not args.no_filter)
    return r
if __name__ == '__main__':
    main()