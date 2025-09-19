from typing import Dict, List, Any
import os
import numpy as np
import matplotlib.pyplot as plt
from cuvisai_examples.registry import REPORTERS

@REPORTERS.register("EfficientADReporter")
class EfficientADReporter:
    def __init__(self, out_dir: str, plot_thresholds: List[float] = None, overlay: str = "RGB"):
        self.out_dir = out_dir
        self.plot_thresholds = plot_thresholds or [0.2, 0.4, 0.6, 0.8]
        self.overlay = overlay

    def save_images(self, batches: List[Dict[str, Any]], predictions: List[Dict[str, Any]]):
        os.makedirs(self.out_dir, exist_ok=True)
        for i, (batch, pred) in enumerate(zip(batches, predictions)):
            img = batch["image"].squeeze().detach().cpu().permute(1, 2, 0).numpy() if hasattr(batch["image"], "detach") else np.asarray(batch["image"])
            a_map = pred["anomaly_map"]
            a_map = a_map + abs(a_map.min())
            fig_rows = 2 + len(self.plot_thresholds)
            fig, ax = plt.subplots(fig_rows, 2, tight_layout=True, dpi=300, figsize=(8, 4 + len(self.plot_thresholds)))
            rgb = img[:, :, :3] if img.shape[2] >= 3 else np.repeat(img[:, :, :1], 3, axis=2)
            ir = img[:, :, 3:] if img.shape[2] > 3 else rgb
            rgb = rgb[:, :, [2, 1, 0]]
            ax[0][0].imshow((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
            ax[0][0].set_title("RGB")
            ax[0][1].imshow((np.clip(ir, 0, 1) * 255).astype(np.uint8))
            ax[0][1].set_title("IR")
            ax[1][0].imshow((np.clip(a_map, 0, None) / (np.max(a_map) + 1e-8)) * 255, vmin=0, vmax=255)
            ax[1][0].set_title("anomaly_map")
            base = rgb.copy() if self.overlay == "RGB" else ir.copy()
            ov = base.copy()
            mask = a_map > min(self.plot_thresholds)
            if ov.ndim == 2:
                ov = np.stack([ov] * 3, axis=2)
            ov[mask, 1] = np.clip(a_map[mask], 0, 1)
            ax[1][1].imshow((np.clip(ov, 0, 1) * 255).astype(np.uint8))
            ax[1][1].set_title("overlay")
            for r, thr in enumerate(self.plot_thresholds):
                thr_map = (a_map >= thr).astype(np.uint8)
                ax[2 + r][0].imshow(thr_map * 255, vmin=0, vmax=255)
                ax[2 + r][0].set_title(f"threshold={thr}")
                ov2 = base.copy()
                if ov2.ndim == 2:
                    ov2 = np.stack([ov2] * 3, axis=2)
                ov2[thr_map == 1] = [1, 0, 0]
                ax[2 + r][1].imshow((np.clip(ov2, 0, 1) * 255).astype(np.uint8))
                ax[2 + r][1].set_title("overlay thr")
            name = batch.get("name", f"sample_{i}")
            fig.suptitle(str(name))
            plt.savefig(os.path.join(self.out_dir, f"{name}.png"))
            plt.close(fig)

    def save_metrics(self, metrics: Dict[str, Any]):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "metrics.json"), "w") as f:
            import json
            json.dump(metrics, f, indent=2)
