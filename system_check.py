import os
import math
import time
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tiktoken
from datasets import load_dataset
import matplotlib.pyplot as plt

print(f'All imports ready')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')