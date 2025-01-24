from onnxruntime.training import artifacts
import os
import onnx

onnx_model = onnx.load(f"data/model.onnx")

# Define the parameters that require their gradients to be computed
# (trainable parameters) and those that do not (frozen/non trainable parameters).
requires_grad = ["classifier.1.weight", "classifier.1.bias"]
frozen_params = [
   param.name
   for param in onnx_model.graph.initializer
   if param.name not in requires_grad
]

artifacts.generate_artifacts(
    onnx_model,
    optimizer=artifacts.OptimType.AdamW,
    loss=artifacts.LossType.CrossEntropyLoss,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
    artifact_directory="data",
    additional_output_names=["output"])