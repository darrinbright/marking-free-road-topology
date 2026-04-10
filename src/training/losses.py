import torch
import torch.nn as nn
import torch.nn.functional as F

class HTGNNLoss(nn.Module):

    def __init__(self, junction_weight: float=4.0, edge_weight: float=0.8, temporal_edge_weight: float=0.4):
        super().__init__()
        self.junction_weight = junction_weight
        self.edge_weight = edge_weight
        self.temporal_edge_weight = temporal_edge_weight

    def _edge_loss(self, edge_logits, neg_edge_logits, node_feats, edge_index):
        if edge_logits.shape[0] == 0:
            return torch.tensor(0.0, device=edge_logits.device if edge_logits.numel() > 0 else node_feats.device)
        edge_tgt = torch.ones_like(edge_logits)
        is_junc = (node_feats[:, 3] > 0.5).float()
        touches_junc = (is_junc[edge_index[0]] > 0.5) | (is_junc[edge_index[1]] > 0.5)
        w = 1.0 + (self.junction_weight - 1.0) * touches_junc.float()
        pos_loss = (F.binary_cross_entropy_with_logits(edge_logits, edge_tgt, reduction='none') * w).mean()
        neg_loss = torch.tensor(0.0, device=edge_logits.device)
        if neg_edge_logits is not None and neg_edge_logits.shape[0] > 0:
            neg_tgt = torch.zeros_like(neg_edge_logits)
            neg_loss = F.binary_cross_entropy_with_logits(neg_edge_logits, neg_tgt, reduction='mean')
        return pos_loss + neg_loss

    def forward(self, preds, targets):
        seg_target = targets['bev_mask']
        seg_pred = F.interpolate(preds['seg_logits'], size=seg_target.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = F.cross_entropy(seg_pred, seg_target.squeeze(1))
        node_feats = targets['node_feats']
        edge_index = targets['edge_index']
        neg_edge_logits = preds.get('neg_edge_logits')
        edge_loss = self._edge_loss(preds['edge_logits'], neg_edge_logits, node_feats, edge_index)
        temporal_loss = torch.tensor(0.0, device=seg_pred.device)
        if 'temporal_edge_logits' in preds and preds['temporal_edge_logits'].shape[0] > 0:
            temporal_loss = self._edge_loss(preds['temporal_edge_logits'], neg_edge_logits, node_feats, edge_index)
        total = seg_loss + self.edge_weight * edge_loss + self.temporal_edge_weight * temporal_loss
        return (total, {'seg': seg_loss.item(), 'edge': edge_loss.item(), 'temporal_edge': temporal_loss.item()})