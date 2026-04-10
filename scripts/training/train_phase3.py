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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ann_dir = ROOT / 'data' / 'annotations'
    idd_root = ROOT / 'data' / 'idd_raw'
    gnn_ckpt = ROOT / 'checkpoints' / 'best_gnn.pth'
    if not gnn_ckpt.exists():
        return
    ds = IDDTopoDataset(str(ann_dir), str(idd_root), augment=False)
    n = len(ds)
    train_ds, val_ds, _ = random_split(ds, [n - 45, 30, 15], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
    model = HTGNN(backbone_ckpt=None).to(device)
    model.load_state_dict(torch.load(gnn_ckpt, map_location=device, weights_only=True))
    model.unfreeze_backbone_partial()
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    gnn_params = list(model.gnn.parameters()) + list(model.memory.parameters())
    opt = torch.optim.AdamW([{'params': backbone_params, 'lr': 1e-05}, {'params': gnn_params, 'lr': 0.0001}], weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    criterion = HTGNNLoss()
    ckpt_dir = ROOT / 'checkpoints'
    best = float('inf')
    for epoch in range(20):
        model.train()
        model.memory.reset()
        tloss = 0
        for b in tqdm(train_loader, desc=f'P3-E{epoch}', file=sys.stdout, dynamic_ncols=True, mininterval=1):
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            preds = model(b)
            loss, _ = criterion(preds, b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item()
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
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), ckpt_dir / 'best_gnn.pth')
if __name__ == '__main__':
    main()