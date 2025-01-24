import os
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import shutil

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset with transformations
for set in ["train_dataset", "test_dataset"]:
    temp_directory = "temp"
    os.makedirs(temp_directory, exist_ok=True)
    mnist_dataset = datasets.MNIST(
        root=temp_directory, 
        train= True if set == "train_dataset" else False, 
        download=True, 
        transform=transform
    )

    # Directory to save preprocessed data
    processed_dir = "data"
    os.makedirs(processed_dir, exist_ok=True)

    # Preprocess and save data
    processed_images = []
    processed_labels = []

    print("Process the dataset")
    for i in tqdm(range(len(mnist_dataset))):
        image, label = mnist_dataset[i]  # Transformation applied here
        processed_images.append(image)
        processed_labels.append(label)

    # Save as tensors
    torch.save((torch.stack(processed_images), torch.tensor(processed_labels)), 
            os.path.join(processed_dir, f"{set}.pt"))


    print(f"Transformed dataset saved to '{processed_dir}/{set}.pt'")

    if os.path.exists(temp_directory):
        # Remove the directory and its contents
        shutil.rmtree(temp_directory)
        print(f"Directory '{temp_directory}' and its contents have been removed.")
    else:
        print(f"Directory '{temp_directory}' does not exist.")

