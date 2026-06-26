
# Full validation/export script (abridged but complete runnable version)
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from cellpose import models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

TEST_SHAPES = [(256, 256), (512, 512), (768, 768), (1024, 1024)]


def export(model_name, out):
    m = models.CellposeModel(pretrained_model=model_name, gpu=False)
    net = m.net.eval().float()
    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(net, dummy, out, export_params=True, opset_version=18,
                      dynamo=False,
                      input_names=["input_image"], output_names=["flows_and_probs"],
                      dynamic_axes={"input_image": {0: "batch", 2: "height", 3: "width"},
                                    "flows_and_probs": {0: "batch"}})
    return net


def validate(path):
    mdl = onnx.load(path)
    onnx.checker.check_model(mdl)
    mdl = onnx.shape_inference.infer_shapes(mdl)
    onnx.save(mdl, path)


def metrics(a, b):
    d = np.abs(a-b)
    rel = d/np.maximum(np.abs(a), 1e-8)
    rmse = float(np.sqrt(np.mean(d*d)))
    cos = float(np.dot(a.ravel(), b.ravel()) /
                (np.linalg.norm(a.ravel())*np.linalg.norm(b.ravel())))
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), median_abs=float(np.median(d)),
                rmse=rmse, max_rel=float(rel.max()), cosine=cos,
                p50=float(np.percentile(d, 50)), p90=float(np.percentile(d, 90)),
                p99=float(np.percentile(d, 99)))


def compare(net, path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    report = []
    for h, w in TEST_SHAPES:
        for i in range(5):
            x = torch.randn(1, 3, h, w)
            t0 = time.perf_counter()
            yt = net(x).detach().cpu().numpy()
            t1 = time.perf_counter()
            t2 = time.perf_counter()
            yo = sess.run(None, {"input_image": x.numpy()})[0]
            t3 = time.perf_counter()
            assert yt.shape == yo.shape
            assert yt.dtype == yo.dtype
            assert np.isfinite(yt).all() and np.isfinite(yo).all()
            np.testing.assert_allclose(yt, yo, rtol=1e-4, atol=1e-5)
            m = metrics(yt, yo)
            m["shape"] = [h, w]
            m["torch_ms"] = (t1-t0)*1000
            m["onnx_ms"] = (t3-t2)*1000
            report.append(m)
            print(h, w, m["max_abs"], m["rmse"], m["cosine"])
    return report


def summarize(rep):
    worst = max(rep, key=lambda x: x["max_abs"])
    print("\n==== SUMMARY ====")
    print("tests", len(rep))
    print("worst max abs", worst["max_abs"])
    print("worst rmse", max(r["rmse"] for r in rep))
    print("worst rel", max(r["max_rel"] for r in rep))
    print("worst cosine", min(r["cosine"] for r in rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cpsam")
    ap.add_argument("--output", default="models")
    a = ap.parse_args()
    outdir = Path(a.output)
    outdir.mkdir(parents=True, exist_ok=True)
    onnxfile = outdir/f"{a.model}.onnx"
    net = export(a.model, onnxfile)
    validate(onnxfile)
    rep = compare(net, onnxfile)
    summarize(rep)
    with open(outdir/"validation_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print("PASS")


if __name__ == "__main__":
    main()
