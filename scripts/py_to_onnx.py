import argparse
from pathlib import Path

import torch
from cellpose import models


def main():

    parser = argparse.ArgumentParser(
        description="Export a Cellpose model to ONNX."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="cpsam",
        help="Name of the pretrained Cellpose model (default: cpsam)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory)",
    )

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    outfile = args.output / f"{args.model}.onnx"

    print(f"Loading model '{args.model}'...")

    model = models.CellposeModel(
        pretrained_model=args.model,
        gpu=False,
    )

    net = model.net.eval().float()

    # CPSAM expects 256x256 inputs
    dummy_input = torch.randn(
        1,
        3,
        256,
        256,
        dtype=torch.float32,
    )

    print(f"Exporting to {outfile}...")

    torch.onnx.export(
        net,
        dummy_input,
        outfile,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["flows_and_probs"],
    )

    print(f"✓ Model exported successfully to '{outfile}'")


if __name__ == "__main__":
    main()
