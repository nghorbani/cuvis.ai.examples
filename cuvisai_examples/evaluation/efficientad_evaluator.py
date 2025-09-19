from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc
from cuvisai_examples.registry import EVALUATORS

@EVALUATORS.register("EfficientADEvaluator")
class EfficientADEvaluator:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def _flatten_binary(self, score_maps: List[np.ndarray], gt_masks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        ys, yt = [], []
        for s, g in zip(score_maps, gt_masks):
            if self.normalize:
                smin, smax = float(np.min(s)), float(np.max(s))
                denom = smax - smin
                s = (s - smin) / (denom + 1e-8) if denom > 1e-8 else np.zeros_like(s)
            g = (g > 0).astype(np.uint8)
            ys.append(s.reshape(-1))
            yt.append(g.reshape(-1))
        return np.concatenate(ys), np.concatenate(yt)

    def evaluate(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        score_maps = [p["anomaly_map"] if isinstance(p["anomaly_map"], np.ndarray) else np.array(p["anomaly_map"]) for p in predictions]
        gt_masks = [p["mask"] if isinstance(p["mask"], np.ndarray) else np.array(p["mask"]) for p in predictions]
        y_scores, y_true = self._flatten_binary(score_maps, gt_masks)
        if len(np.unique(y_true)) < 2:
            return {"overall_auc": float("nan")}
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        return {"overall_auc": float(roc_auc)}
