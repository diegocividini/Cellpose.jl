import torch
from cellpose import models

model_name = 'cpsam'
model = models.CellposeModel(pretrained_model=model_name, gpu=False)

pytorch_net = model.net
pytorch_net.eval()
pytorch_net = pytorch_net.float()

# cpsam wants a fixed input size of 256x256
dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32) 

# ONNX export
torch.onnx.export(
    pytorch_net, 
    dummy_input, 
    f"{model_name}.onnx", 
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input_image'], 
    output_names=['flows_and_probs']
)

print(f"Model {model_name} ONNX exported successfully to 256x256!")