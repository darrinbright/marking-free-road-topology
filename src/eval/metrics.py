import numpy as np
from sklearn.metrics import jaccard_score

def seg_iou(pred_mask, gt_mask):
    p = (pred_mask > 0).flatten()
    g = (gt_mask > 0).flatten()
    return jaccard_score(g, p, average='binary', zero_division=0)