import sys
import torch
import transformers
from datasets import load_dataset

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test dataset loading
print("Testing dataset loading...")
try:
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:5]")
    print(f"✅ Successfully loaded {len(dataset)} samples")
    print("Sample:", dataset[0])
except Exception as e:
    print(f"❌ Error loading dataset: {e}")