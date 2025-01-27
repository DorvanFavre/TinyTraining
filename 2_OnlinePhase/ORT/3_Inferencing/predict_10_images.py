from onnxruntime import InferenceSession
import os
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np


test_dataset_path = os.path.join("data", "test_dataset.pt")

test_images, test_labels = torch.load(test_dataset_path, weights_only=False)

class myDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create Dataset and DataLoader
test_dataset = myDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

session = InferenceSession('data/inference_model.onnx',providers=['CPUExecutionProvider'])

# Perform 10 inferences

plt.figure(figsize=(50, 20))
for i, (image, label) in enumerate(test_loader):
    if i >= 10:  # Limit to 10 inferences
        break
    
    # Preprocess the input (if required by the model)
    input_image = image.numpy().astype(np.float32)
    print(input_image.shape)
    
    # Get the input name for the model
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: input_image})
    
    # Extract the model prediction
    prediction = np.argmax(outputs[0], axis=1)

    # Remove the batch dimension for visualization
    image_to_plot = np.squeeze(input_image, axis=0)  # Now shape is (3, 128, 128)

    # If the image is in CHW format (Channels, Height, Width), transpose to HWC format for visualization
    image_to_plot = np.transpose(image_to_plot, (1, 2, 0)) 
    
    # Plot the image and results
    plt.subplot(10, 1, i + 1)
    plt.imshow(image_to_plot, cmap="gray")
    plt.title(f"True Label: {label.item()}, Predicted: {prediction[0]}")
plt.axis("off")
plt.subplots_adjust(hspace=1.0)
plt.savefig("data/output.png")
plt.show()


    