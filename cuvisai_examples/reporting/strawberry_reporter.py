import json
import os
from cuvisai_examples.registry import REPORTERS


@REPORTERS.register("StrawberryReporter")
class StrawberryReporter:
    def __init__(self, out_dir: str | None = None):
        self.out_dir = out_dir

    def save(self, metrics, outputs):
        out_dir = self.out_dir or outputs.get("out_dir", "./work_dirs/exp/reports")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    def save_metrics(self, metrics):
        out_dir = self.out_dir or "./work_dirs/exp/reports"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
