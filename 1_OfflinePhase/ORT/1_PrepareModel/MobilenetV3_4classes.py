'''
MobilenetV2_4classes.py

4 classes classification with a pretrained MobilenetV2 model from torchvision with IMAGENET1K_V2 wheights.
Remplace the last layer in order to have 4 ouputs. 
'''

import torchvision
import torch
import os

model = torchvision.models.mobilenet_v3_small(
   weights=torchvision.models.MobileNet_V3_Small_Weights)

print(model.graph)

# The original model is trained on imagenet which has 1000 classes.
# For our image classification scenario, we need to classify among 4 categories.
# So we need to change the last layer of the model to have 4 outputs.
model.classifier[1] = torch.nn.Linear(1024, 4)

# Export the model to ONNX.
os.makedirs("data", exist_ok=True)
model_name = "model"
torch.onnx.export(model, torch.randn(1, 3, 128, 128),
                  f"data/{model_name}.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})