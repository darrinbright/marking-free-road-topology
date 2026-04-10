import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.data.dataset import IDDTopoDataset
from src.models.backbone import BEVBackbone

def collate(batch):
    return batch[0]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ann_dir = ROOT / 'data' / 'annotations'
    idd_root = ROOT / 'data' / 'idd_raw'
    ds = IDDTopoDataset(str(ann_dir), str(idd_root), augment=True)
    n = len(ds)
    train_ds, val_ds, test_ds = random_split(ds, [n - 45, 30, 15], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
    model = BEVBackbone().to(device)
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.encoder.encoder.block[-2:].parameters():
        p.requires_grad = True
    for p in model.fuse.parameters():
        p.requires_grad = True
    for p in model.seg_head.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-05, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    epochs = 30
    ckpt_dir = ROOT / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    best = float('inf')
    for epoch in range(epochs):
        model.train()
        tloss = 0
        for b in tqdm(train_loader, desc=f'E{epoch}', file=sys.stdout, dynamic_ncols=True, mininterval=1):
            bev_img = b['bev_img'].to(device)
            bev_mask = b['bev_mask'].to(device)
            _, seg_logits = model(bev_img)
            seg_pred = F.interpolate(seg_logits, size=bev_mask.shape[1:], mode='bilinear', align_corners=False)
            loss = F.cross_entropy(seg_pred, bev_mask.squeeze(1))
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
                bev_img = b['bev_img'].to(device)
                bev_mask = b['bev_mask'].to(device)
                _, seg_logits = model(bev_img)
                seg_pred = F.interpolate(seg_logits, size=bev_mask.shape[1:], mode='bilinear', align_corners=False)
                loss = F.cross_entropy(seg_pred, bev_mask.squeeze(1))
                vloss += loss.item()
        vloss /= len(val_loader)
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), ckpt_dir / 'best_backbone.pth')
if __name__ == '__main__':
    main()