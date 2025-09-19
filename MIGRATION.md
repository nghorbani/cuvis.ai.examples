Migration notes

- Old YAMLs map to Hydra/OmegaConf configs under cuvisai_examples/configs.
- Key mapping:
  - Datasets: use dataset/*.yaml with type: and params:
  - Models: use model/* with type: and params:
  - Runner/trainer settings moved under runner/lightning.yaml and train.yaml trainer.*
- Legacy scripts:
  - Replace direct scripts with:
    - cuvisai-train
    - cuvisai-infer
    - cuvisai-report
- Minimal example:
  cuvisai-train
  (uses placeholder configs; replace with efficientad/perpixel_ae/strawberry-specific configs in subsequent commits)
