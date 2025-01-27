import os
import torch
import matplotlib.pyplot as plt

print("Load datasets...")
# Path to the preprocessed dataset
train_dataset_path = os.path.join("data", "train_dataset.pt")
test_dataset_path = os.path.join("data", "test_dataset.pt")

# Load the preprocessed dataset
train_images, train_labels = torch.load(train_dataset_path, weights_only=False)
test_images, test_labels = torch.load(test_dataset_path, weights_only=False)

print(f"Loaded train images shape: {train_images.shape}, Labels shape: {train_labels.shape}")
print(f"Loaded test images shape: {test_images.shape}, Labels shape: {test_labels.shape}")

# Function to display a few images from the dataset
def show_images(images, labels, num_images=5, title="Dataset Examples"):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        image = images[i]
        if image.dim() == 3:  # Assuming (C, H, W) format
            plt.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        elif image.dim() == 2:  # Assuming (H, W) format
            plt.imshow(image.cpu().numpy(), cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.savefig("data/output.png")
    print("Images saved in data folder as output.png")

# Display a few images from the training dataset
show_images(train_images, train_labels, num_images=5, title="Train Dataset Examples")

# Display a few images from the test dataset
show_images(test_images, test_labels, num_images=5, title="Test Dataset Examples")
