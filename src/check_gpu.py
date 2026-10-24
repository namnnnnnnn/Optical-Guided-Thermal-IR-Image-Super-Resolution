"""
check_gpu.py
Role: Environment check - prints PyTorch / CUDA versions and the GPU name.
"""

import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))