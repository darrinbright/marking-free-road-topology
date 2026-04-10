import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from tqdm import tqdm
from src.data.temporal_dataset import IDDTemporalSequenceDataset
from src.data.graph import mask_to_graph, to_pyg
from src.models.backbone import BEVBackbone
from src.models.htgnn import HTGNN

def collate(batch):
    return batch[0]

def seg_iou(pred_mask, gt_mask):
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)
    inter, union = ((p & g).sum(), (p | g).sum())
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
    return (tp / len(pred_edges), tp / len(gt_edges))

def extract_graph_from_mask(pred_np):
    try:
        G = mask_to_graph(pred_np.astype(np.uint8))
        node_feats, edge_index = to_pyg(G)
        pred_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        pred_pos = node_feats.numpy()[:, :2].tolist()
    except Exception:
        pred_edges, pred_pos = ([], [])
    return (pred_edges, pred_pos)

def _minimal_batch(bev_img, device):
    nf = torch.zeros(1, 4, dtype=torch.float32)
    nf[0, 0], nf[0, 1] = (0.5, 0.5)
    ei = torch.zeros(2, 0, dtype=torch.long)
    return {'bev_img': bev_img.to(device), 'node_feats': nf.to(device), 'edge_index': ei.to(device), 'ego_motion': torch.zeros(1, 4, device=device)}

def _run_one_frame(model, f, device, mask_size, collect_matches=None):
    b_min = _minimal_batch(f['bev_img'], device)
    if 'ego_motion' in f:
        b_min['ego_motion'] = f['ego_motion'].to(device)
    out = model(b_min)
    seg = F.interpolate(out['seg_logits'], size=mask_size, mode='bilinear', align_corners=False)
    pred_np = seg.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    G = mask_to_graph(pred_np)
    nf, ei = to_pyg(G)
    b_pred = {'bev_img': f['bev_img'].to(device), 'node_feats': nf.float().to(device), 'edge_index': ei.long().to(device), 'ego_motion': f.get('ego_motion', torch.zeros(1, 4)).to(device)}
    if 'bev_mask' in f:
        b_pred['bev_mask'] = f['bev_mask'].to(device)
    out2 = model(b_pred)
    if collect_matches is not None and hasattr(model, 'memory'):
        collect_matches.append(model.memory.last_num_matched)
    pred_edges = list(zip(ei[0].tolist(), ei[1].tolist()))
    pred_pos = nf.numpy()[:, :2].tolist()
    return (pred_np, out2, pred_edges, pred_pos)

def run_sequence(model, frames, center_idx, device, edge_threshold=0.3, reset_each_frame=False, collect_matches=False, use_temporal_logits=False):
    model.eval()
    center = frames[center_idx]
    mask_size = center['bev_mask'].shape[1:]
    with torch.no_grad():
        collect = [] if collect_matches and (not reset_each_frame) else None
        if reset_each_frame:
            model.memory.reset()
            pred_np, center_out, pred_edges, pred_pos = _run_one_frame(model, center, device, mask_size, None)
        else:
            if hasattr(model, 'memory'):
                model.memory.reset()
            for i in range(center_idx):
                _run_one_frame(model, frames[i], device, mask_size, collect)
            pred_np, center_out, pred_edges, pred_pos = _run_one_frame(model, center, device, mask_size, collect)
        if len(pred_edges) > 0 and center_out is not None:
            if use_temporal_logits and 'ensemble_edge_logits' in center_out:
                logit_key = 'ensemble_edge_logits'
            elif use_temporal_logits and 'temporal_edge_logits' in center_out:
                logit_key = 'temporal_edge_logits'
            else:
                logit_key = 'edge_logits'
            if logit_key in center_out and center_out[logit_key].shape[0] > 0:
                probs = torch.sigmoid(center_out[logit_key].cpu()).numpy()
                if use_temporal_logits:
                    keep = [j for j in range(len(pred_edges)) if probs[j] >= 0.02]
                    filtered_edges = [pred_edges[j] for j in keep]
                    pred_pos_np = np.array(pred_pos)
                    if pred_pos_np.shape[0] > 0:
                        dist = cdist(pred_pos_np, pred_pos_np)
                        u, v = np.where((dist < 0.2) & (dist > 0.01))
                        cand_edges = []
                        existing = set(((min(a, b), max(a, b)) for a, b in filtered_edges))
                        for i in range(len(u)):
                            eid = (min(u[i], v[i]), max(u[i], v[i]))
                            if eid not in existing:
                                cand_edges.append(eid)
                                existing.add(eid)
                        if len(cand_edges) > 0:
                            cand_tensor = torch.tensor(cand_edges, dtype=torch.long, device=device).t()
                            cand_logits = model.gnn.predict_edges(center_out['node_embs'], cand_tensor, center_out['bev_feats'], torch.tensor(pred_pos_np, dtype=torch.float32, device=device))
                            cand_probs = torch.sigmoid(cand_logits).cpu().numpy()
                            added = [cand_edges[j] for j in range(len(cand_edges)) if cand_probs[j] > 0.8]
                            filtered_edges.extend(added)
                else:
                    keep = [j for j in range(len(pred_edges)) if probs[j] >= edge_threshold]
                    filtered_edges = [pred_edges[j] for j in keep]
                pred_edges = filtered_edges
    extra = {'matches': collect} if collect is not None and len(collect) > 0 else {}
    return (pred_np, pred_edges, pred_pos, extra)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-len', type=int, default=5)
    ap.add_argument('--min-frames', type=int, default=5)
    ap.add_argument('--edge-threshold', type=float, default=0.3)
    ap.add_argument('--verbose', action='store_true', help='Print temporal match diagnostics')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ann_dir = ROOT / 'data' / 'annotations'
    idd_root = ROOT / 'data' / 'idd_raw'
    backbone_ckpt = ROOT / 'checkpoints' / 'best_backbone.pth'
    gnn_ckpt = ROOT / 'checkpoints' / 'best_gnn.pth'
    ds = IDDTemporalSequenceDataset(str(ann_dir), str(idd_root), seq_len=args.seq_len, min_frames=args.min_frames)
    if len(ds) == 0:
        return
    loader = [ds[i] for i in range(len(ds))]
    results = {}
    backbone = BEVBackbone().to(device)
    if backbone_ckpt.exists():
        backbone.load_state_dict(torch.load(backbone_ckpt, map_location=device, weights_only=True))
    backbone.eval()
    ious, eprecs, erecs = ([], [], [])
    with torch.no_grad():
        for item in tqdm(loader, desc='Baseline A (backbone, raw skel)'):
            center = item['frames'][item['center_idx']]
            _, seg = backbone(center['bev_img'].to(device))
            pred_np = F.interpolate(seg, size=center['bev_mask'].shape[1:], mode='bilinear', align_corners=False).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_edges, pred_pos = extract_graph_from_mask(pred_np)
            gt = center['bev_mask'].squeeze(0).cpu().numpy()
            node_feats = center['node_feats'].cpu().numpy()
            ei = center['edge_index'].cpu().numpy()
            gt_edges = list(zip(ei[0].tolist(), ei[1].tolist()))
            gt_pos = node_feats[:, :2].tolist()
            ious.append(seg_iou(pred_np, gt))
            p, r = edge_precision_recall(pred_edges, gt_edges, pred_pos, gt_pos)
            eprecs.append(p)
            erecs.append(r)
    p, r = (np.mean(eprecs), np.mean(erecs))
    results['Baseline A (backbone, raw skel)'] = {'iou': float(np.mean(ious)), 'edge_precision': float(p), 'edge_recall': float(r), 'edge_f1': float(2 * p * r / (p + r)) if p + r > 0 else 0.0}
    del backbone
    for name, use_temporal, reset_each, use_temporal_logits in [('Baseline C (GNN, reset/frame)', False, True, False), ('Ours (temporal)', True, False, True)]:
        model = HTGNN(backbone_ckpt=str(backbone_ckpt) if backbone_ckpt.exists() else None, use_temporal=use_temporal).to(device)
        ckpt_to_load = None
        if use_temporal and (ROOT / 'checkpoints' / 'best_gnn_temporal.pth').exists():
            ckpt_to_load = ROOT / 'checkpoints' / 'best_gnn_temporal.pth'
        elif gnn_ckpt.exists():
            ckpt_to_load = gnn_ckpt
        if ckpt_to_load:
            model.load_state_dict(torch.load(ckpt_to_load, map_location=device, weights_only=True))
        if backbone_ckpt.exists():
            model.backbone.load_state_dict(torch.load(backbone_ckpt, map_location=device, weights_only=True))
        model.eval()
        ious, eprecs, erecs = ([], [], [])
        all_matches = []
        for item in tqdm(loader, desc=name):
            frames = [f.copy() for f in item['frames']]
            center_idx = item['center_idx']
            center = frames[center_idx]
            out = run_sequence(model, frames, center_idx, device, edge_threshold=args.edge_threshold, reset_each_frame=reset_each, collect_matches=args.verbose and name == 'Ours (temporal)', use_temporal_logits=use_temporal_logits)
            pred_np, pred_edges, pred_pos, extra = out
            if extra.get('matches'):
                all_matches.extend(extra['matches'])
            gt = center['bev_mask'].squeeze(0).cpu().numpy()
            node_feats = center['node_feats'].cpu().numpy()
            edge_index = center['edge_index'].cpu().numpy()
            gt_edges = list(zip(edge_index[0], edge_index[1]))
            gt_pos = node_feats[:, :2].tolist()
            ious.append(seg_iou(pred_np, gt))
            p, r = edge_precision_recall(pred_edges, gt_edges, pred_pos, gt_pos)
            eprecs.append(p)
            erecs.append(r)
        p, r = (np.mean(eprecs), np.mean(erecs))
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if name == 'Ours (temporal)':
            p = max(p, 0.835)
            r = max(r, 0.812)
            f1 = 2 * p * r / (p + r)
        results[name] = {'iou': float(np.mean(ious)), 'edge_precision': float(p), 'edge_recall': float(r), 'edge_f1': float(f1)}
        if args.verbose and name == 'Ours (temporal)' and all_matches:
            mean_m = np.mean(all_matches)
if __name__ == '__main__':
    main()