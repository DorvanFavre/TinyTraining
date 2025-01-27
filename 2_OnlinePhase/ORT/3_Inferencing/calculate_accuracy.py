import os
import torch
import numpy as np
from onnxruntime import InferenceSession
from torch.utils.data import Dataset, DataLoader

# Load the test dataset
test_dataset_path = os.path.join("data", "test_dataset.pt")
test_images, test_labels = torch.load(test_dataset_path, weights_only=False)

# Define a Dataset class
class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create a Dataset and DataLoader
test_dataset = MyDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the ONNX model
model_path = "data/inference_model.onnx"
session = InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Get the input name for the model
input_name = session.get_inputs()[0].name

# Initialize counters
correct_predictions = 0
total_samples = 0

# Evaluate the model
for image, label in test_loader:
    # Preprocess the input image
    input_image = image.numpy().astype(np.float32)  # Convert tensor to numpy array

    # Run inference
    outputs = session.run(None, {input_name: input_image})

    # Get the predicted label
    predicted_label = np.argmax(outputs[0], axis=1)

    # Check if the prediction is correct
    if predicted_label[0] == label.item():
        correct_predictions += 1

    total_samples += 1

# Calculate accuracy
accuracy = correct_predictions / total_samples * 100
print(f"Accuracy calculated on {total_samples} images.")
print(f"Accuracy: {accuracy:.2f}%")
