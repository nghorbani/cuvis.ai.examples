import sys
print("Python:", sys.version)
import cuvisai_examples
from cuvisai_examples.registry import DATASETS, MODELS
print("DATASETS:", DATASETS.keys())
print("MODELS:", MODELS.keys())
