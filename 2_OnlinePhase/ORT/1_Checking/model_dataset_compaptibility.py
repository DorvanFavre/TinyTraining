from onnxruntime import InferenceSession
import os
import torch
import numpy as np

# Paths to files
model_path = "data/model.onnx"
dataset_path = "data/train_dataset.pt"

# Load the ONNX model
session = InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Load the dataset
train_images, train_labels = torch.load(dataset_path, weights_only=False)

# Check input details
input_name = session.get_inputs()[0].name
model_input_shape = session.get_inputs()[0].shape  # Model's expected input shape
model_input_shape = [dim if dim is not None else -1 for dim in model_input_shape]  # Replace dynamic dims with -1

# Check shape compatibility with the first image in the dataset
sample_image = train_images[0]  # First image from the dataset
sample_shape = list(sample_image.shape)

# Print shapes for debugging
print(f"Model expected input shape: {model_input_shape}")
print(f"Sample image shape: {sample_shape}")

# Validate compatibility
if len(model_input_shape) != len(sample_shape):
    print("Incompatible: Model expects a different number of dimensions.")
else:
    compatible = all(
        model_dim == sample_dim or model_dim == -1
        for model_dim, sample_dim in zip(model_input_shape, sample_shape)
    )
    if compatible:
        print("Shapes are compatible!")
    else:
        print("Shapes are incompatible!")

# Optionally print detailed feedback
for idx, (model_dim, sample_dim) in enumerate(zip(model_input_shape, sample_shape)):
    print(f"Dimension {idx}: Model expects {model_dim}, sample provides {sample_dim}")