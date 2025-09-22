import sys
from cuvisai_examples.registry import DATASETS, MODELS

print("Python:", sys.version)

print("DATASETS:", DATASETS.keys())
print("MODELS:", MODELS.keys())
