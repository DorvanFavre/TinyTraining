import os
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import shutil
from torch.utils.data import random_split, Subset, DataLoader

os.makedirs("data", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def prepare_dataset(split, file_name):

    print(f"prepare {split} dataset")
    dataset = datasets.Flowers102(
        root="data",
        download=True,
        split="test" if split=="train" else "train", # They have inversed the test and train set -.-'
        transform=transform
    )

    # Select only 4 categories (modify as needed)
    selected_categories = [0, 1, 2, 3]

    # Filter the dataset to include only selected categories
    selected_indices = [
        idx for idx, (_, label) in enumerate(dataset)
        if label in selected_categories
    ]
    filtered_dataset = Subset(dataset, selected_indices)

    # # Define the percentage split for training and test sets
    # train_size = int(0.8 * len(filtered_dataset))  # 80% for training
    # test_size = len(filtered_dataset) - train_size  # 20% for testing

    # Directory to save preprocessed data
    os.makedirs("data", exist_ok=True)

    # # Randomly split the filtered dataset into training and testing sets
    # train_dataset, test_dataset = random_split(filtered_dataset, [train_size, test_size])

    dataloader = DataLoader(filtered_dataset, batch_size=1, shuffle=True)

    # Function to preprocess and save datasets
    processed_images = []
    processed_labels = []
    print(f"Processing {file_name} dataset")
    for image, label in tqdm(dataloader):
            processed_images.append(image.squeeze(0))  # Remove batch dimension
            processed_labels.append(label.item())
    
    # Save as tensors
    torch.save(
        (torch.stack(processed_images), torch.tensor(processed_labels)),
        os.path.join("data", file_name)
    )
    print(f"Transformed dataset saved to '{os.path.join('data', file_name)}'")

        # Print the number of samples in the original dataset
    print(f"Original dataset size: {len(dataset)}")

    # Print the number of samples in the filtered subset dataset
    print(f"Subset dataset size (selected categories): {len(filtered_dataset)}")

# Preprocess and save the training and testing datasets
prepare_dataset("train", "train_dataset.pt")
prepare_dataset("test", "test_dataset.pt")

