from onnxruntime.training.api import CheckpointState, Module, Optimizer
import numpy as np
import evaluate
import torch
import os
from tqdm import tqdm


num_epochs = 5

print("Load datasets...")
# Path to the preprocessed dataset
train_dataset_path = os.path.join("data", "train_dataset.pt")
test_dataset_path = os.path.join("data", "test_dataset.pt")

# Load the preprocessed dataset
train_images, train_labels = torch.load(train_dataset_path, weights_only=False)
test_images, test_labels = torch.load(test_dataset_path, weights_only=False)

print(f"Loaded train images shape: {train_images.shape}, Labels shape: {train_labels.shape}")
print(f"Loaded test images shape: {test_images.shape}, Labels shape: {test_labels.shape}")

from torch.utils.data import Dataset, DataLoader

class myDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create Dataset and DataLoader
train_dataset = myDataset(train_images, train_labels)
test_dataset = myDataset(test_images, test_labels)
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)



# Create checkpoint state.
state = CheckpointState.load_checkpoint("data/checkpoint")

# Create module.
model = Module("data/training_model.onnx", state, "data/eval_model.onnx")

# Create optimizer.
optimizer = Optimizer("data/optimizer_model.onnx", model)

# Util function to convert logits to predictions.
def get_pred(logits):
    return np.argmax(logits, axis=1)

# Training Loop :
def train(epoch):
    model.train()
    losses = []
    for _, (data, target) in tqdm(enumerate(train_loader)):
        forward_inputs = [data.numpy(),target.numpy().astype(np.int64)]
        train_loss, _ = model(*forward_inputs)
        optimizer.step()
        model.lazy_reset_grad()
        losses.append(train_loss)

    print(f'Epoch: {epoch+1}/ {num_epochs}, Train Loss: {sum(losses)/len(losses):.4f}')

# Test Loop :
def test(epoch):
    model.eval()
    losses = []
    metric = evaluate.load('accuracy')

    for _, (data, target) in tqdm(enumerate(train_loader)):
        forward_inputs = [data.numpy(),target.numpy().astype(np.int64)]
        test_loss, logits = model(*forward_inputs)
        metric.add_batch(references=target, predictions=get_pred(logits))
        losses.append(test_loss)

    metrics = metric.compute()
    print(f'Epoch: {epoch+1}, Test Loss: {sum(losses)/len(losses):.4f}, Accuracy : {metrics["accuracy"]:.2f}')



for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("train")
    train(epoch)
    print("test")
    test(epoch)

model.export_model_for_inferencing("data/inference_model.onnx",["output"])
print("Training done")