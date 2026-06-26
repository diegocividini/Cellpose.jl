import torch
import sys

# 🔥 FIX CHIRURGICO PER PYTORCH NIGHTLY E PYTHON 3.14 🔥
# Blocking ONNXScript rule that causes an error when exporting to ONNX
try:
    import onnxscript.rewriter.rules.common._remove_optional_bias as opt_bias
    opt_bias.check = lambda *args, **kwargs: False
    print("✅ ONNXScript bug bypassed successfully!")
except Exception as e:
    print(
        f"Error occurred while trying to bypass ONNXScript bug (maybe not needed in this build): {e}")

from cellpose import models

model_name = 'cpsam'
model = models.CellposeModel(pretrained_model=model_name, gpu=False)

pytorch_net = model.net
pytorch_net.eval()
pytorch_net = pytorch_net.float()

dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

print("Exporting model to ONNX...")
# ONNX export (senza toccare nulla, il bug è stato neutralizzato sopra)
torch.onnx.export(
    pytorch_net,
    dummy_input,
    f"{model_name}.onnx",
    export_params=True,
    opset_version=17,
    input_names=['input_image'],
    output_names=['flows_and_probs']
)

print(f"Model {model_name} ONNX exported successfully to 256x256!")
