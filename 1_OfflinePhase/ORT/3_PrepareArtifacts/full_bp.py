from onnxruntime.training import artifacts
import os
import onnx

'''
The traning technic consiste of retraning the entire model, so we need to set all the parameters to be trainable.
'''

onnx_file_path = "data/model.onnx"
onnx_model = onnx.load(onnx_file_path)

requires_grad = [param.name for param in onnx_model.graph.initializer]
print(f"Requires grad: {requires_grad}")

frozen_params = []
print(f"Frozen params: {frozen_params}")

artifacts.generate_artifacts(
    onnx_model,
    optimizer=artifacts.OptimType.AdamW,
    loss=artifacts.LossType.CrossEntropyLoss,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
    artifact_directory="data",
    additional_output_names=["output"])