import argparse
import os
import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.data.dataset import IDDTopoDataset
from src.models.htgnn import HTGNN
from src.training.losses import HTGNNLoss

def collate(batch):
    return batch[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    args = ap.parse_args()
    use_wandb = args.wandb
    try:
        if use_wandb:
            import wandb
            wandb.init(project='htgnn', config={'phase': 'gnn', 'neg_edges': True})
    except ImportError:
        use_wandb = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ann_dir = ROOT / 'data' / 'annotations'
    idd_root = ROOT / 'data' / 'idd_raw'
    ds = IDDTopoDataset(str(ann_dir), str(idd_root), augment=False)
    n = len(ds)
    train_ds, val_ds, test_ds = random_split(ds, [n - 45, 30, 15], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
    backbone_ckpt = ROOT / 'checkpoints' / 'best_backbone.pth'
    model = HTGNN(backbone_ckpt=str(backbone_ckpt) if backbone_ckpt.exists() else None).to(device)
    for p in model.backbone.encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0003, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    criterion = HTGNNLoss()
    ckpt_dir = ROOT / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    best = float('inf')
    for epoch in range(50):
        model.train()
        model.memory.reset()
        tloss = 0
        for step, b in enumerate(tqdm(train_loader, desc=f'E{epoch}', file=sys.stdout, dynamic_ncols=True, mininterval=1)):
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            preds = model(b)
            loss, ld = criterion(preds, b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item()
            if use_wandb and step % 10 == 0:
                import wandb
                wandb.log({'train/seg_loss': ld['seg'], 'train/edge_loss': ld['edge'], 'train/total': loss.item()}, step=epoch * len(train_loader) + step)
        sched.step()
        model.eval()
        model.memory.reset()
        vloss = 0
        with torch.no_grad():
            for b in val_loader:
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                preds = model(b)
                loss, _ = criterion(preds, b)
                vloss += loss.item()
        vloss /= len(val_loader)
        if use_wandb:
            import wandb
            wandb.log({'val/loss': vloss, 'val/epoch': epoch}, step=(epoch + 1) * len(train_loader))
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), ckpt_dir / 'best_gnn.pth')
if __name__ == '__main__':
    main()