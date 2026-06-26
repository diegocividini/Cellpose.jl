import torch
from cellpose import models
import torch.onnx

model_name = 'cpsam'
model = models.CellposeModel(pretrained_model=model_name, gpu=False)

pytorch_net = model.net
pytorch_net.eval()
pytorch_net = pytorch_net.float()

dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

print("Exporting model to ONNX using legacy-compatible config...")

torch.onnx.export(
    pytorch_net,
    dummy_input,
    f"{model_name}.onnx",
    export_params=True,
    opset_version=17,
    input_names=['input_image'],
    output_names=['flows_and_probs'],
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX
)

print(f"Model {model_name} ONNX exported successfully!")
