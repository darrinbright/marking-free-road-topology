import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.data.temporal_dataset import IDDTemporalSequenceDataset
from src.data.dataset import _sample_negative_edges
from src.data.graph import mask_to_graph, to_pyg
from src.models.htgnn import HTGNN
from src.training.losses import HTGNNLoss

def collate(batch):
    return batch[0]

def _minimal_batch(bev_img, device):
    nf = torch.zeros(1, 4, dtype=torch.float32)
    nf[0, 0], nf[0, 1] = (0.5, 0.5)
    return {'bev_img': bev_img.to(device), 'node_feats': nf.to(device), 'edge_index': torch.zeros(2, 0, dtype=torch.long, device=device), 'ego_motion': torch.zeros(1, 4, device=device)}

def run_sequence_train(model, item, device, criterion):
    frames = item['frames']
    center_idx = item['center_idx']
    center = frames[center_idx]
    model.memory.reset()
    with torch.no_grad():
        for i in range(center_idx):
            f = frames[i]
            bev_img = f['bev_img'].to(device)
            b_min = _minimal_batch(bev_img, device)
            if 'ego_motion' in f:
                b_min['ego_motion'] = f['ego_motion'].to(device)
            model(b_min)
    bev_img = center['bev_img'].to(device)
    bev_mask = center['bev_mask'].to(device)
    node_feats = center['node_feats'].float().to(device)
    edge_index = center['edge_index'].long().to(device)
    neg_edge_index = _sample_negative_edges(edge_index.cpu(), node_feats.shape[0], node_feats[:, :2].cpu()).to(device)
    b_center = {'bev_img': bev_img, 'node_feats': node_feats, 'edge_index': edge_index, 'neg_edge_index': neg_edge_index, 'bev_mask': bev_mask, 'ego_motion': center.get('ego_motion', torch.zeros(1, 4)).to(device)}
    preds = model(b_center)
    loss, ld = criterion(preds, b_center)
    return (loss, ld)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ann_dir = ROOT / 'data' / 'annotations'
    idd_root = ROOT / 'data' / 'idd_raw'
    backbone_ckpt = ROOT / 'checkpoints' / 'best_backbone.pth'
    gnn_ckpt = ROOT / 'checkpoints' / 'best_gnn.pth'
    ds = IDDTemporalSequenceDataset(str(ann_dir), str(idd_root), seq_len=5, min_frames=3)
    if len(ds) == 0:
        return
    n = len(ds)
    n_train = max(1, int(0.8 * n))
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
    model = HTGNN(backbone_ckpt=str(backbone_ckpt) if backbone_ckpt.exists() else None, use_temporal=True).to(device)
    if gnn_ckpt.exists():
        model.load_state_dict(torch.load(gnn_ckpt, map_location=device, weights_only=True))
    for p in model.backbone.encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    criterion = HTGNNLoss()
    ckpt_dir = ROOT / 'checkpoints'
    best_val = float('inf')
    for epoch in range(20):
        model.train()
        tloss = 0
        for b in tqdm(train_loader, desc=f'E{epoch}', file=sys.stdout, dynamic_ncols=True, mininterval=1):
            loss, _ = run_sequence_train(model, b, device, criterion)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item()
        sched.step()
        model.eval()
        vloss = 0
        with torch.no_grad():
            for b in val_loader:
                loss, _ = run_sequence_train(model, b, device, criterion)
                vloss += loss.item()
        vloss /= max(len(val_loader), 1)
        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), ckpt_dir / 'best_gnn_temporal.pth')
if __name__ == '__main__':
    main()