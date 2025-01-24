import torch
import os

# Pytorch class that we will use to generate the graphs.
class MNISTNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MNISTNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, model_input):
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Create a MNISTNet instance.
device = "cpu"
batch_size, input_size, hidden_size, output_size = 64, 784, 500, 10
pt_model = MNISTNet(input_size, hidden_size, output_size).to(device)

onnx_file_path = "data/model.onnx"

# Generate a random input.
model_inputs = (torch.randn(batch_size, input_size, device=device),)

model_outputs = pt_model(*model_inputs)
if isinstance(model_outputs, torch.Tensor):
    model_outputs = [model_outputs]
    
input_names = ["input"]
output_names = ["output"]
dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

os.makedirs("data", exist_ok=True)
torch.onnx.export(
    pt_model,
    model_inputs,
    onnx_file_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=14,
    do_constant_folding=False,
    training=torch.onnx.TrainingMode.TRAINING,
    dynamic_axes=dynamic_axes,
    export_params=True,
    keep_initializers_as_inputs=False,
)