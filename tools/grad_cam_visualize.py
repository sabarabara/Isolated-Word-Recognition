#!/usr/bin/env python3
"""Generate Grad-CAM for any registered model strategy.

Examples:
  python tools/grad_cam_visualize.py \
      --model-type conformer \
      --checkpoint results/conformer/checkpoints/best_model.pt \
      --sample-index 0

  python tools/grad_cam_visualize.py \
      --model-type cnn2d \
      --checkpoint results/cnn2d/checkpoints/best_model.pt \
      --layer conv_layers.10 \
      --sample-index 3 \
      --target-class 5
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factories import create_model_strategy
import strategies.model_strategy.strategys as model_mods
import strategies.evaluation_strategy.strategies as eval_mods


def load_plugins() -> None:
    for module_root in [model_mods, eval_mods]:
        for _loader, name, _is_pkg in pkgutil.walk_packages(
            module_root.__path__, module_root.__name__ + "."
        ):
            try:
                importlib.import_module(name)
            except Exception as exc:
                print(f"[load_plugins] skipped '{name}': {exc}")


def resolve_module(model: nn.Module, module_path: str) -> nn.Module:
    module: Any = model
    for token in module_path.split("."):
        if token.isdigit():
            module = module[int(token)]
        else:
            module = getattr(module, token)
    if not isinstance(module, nn.Module):
        raise TypeError(f"Resolved object is not nn.Module: {module_path}")
    return module


def find_last_conv_layer(model: nn.Module) -> tuple[str, nn.Module]:
    candidates: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            candidates.append((name, module))
    if not candidates:
        raise ValueError("No Conv layer found. Provide --layer explicitly for CAM target.")
    return candidates[-1]


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _input, output):
        self.activations = output

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, x: torch.Tensor, target_class: int | None = None) -> dict[str, Any]:
        logits = self.model(x)
        if logits.ndim != 2:
            raise ValueError(f"Expected classifier logits shape (B, C), got {tuple(logits.shape)}")

        pred_class = int(logits.argmax(dim=1).item())
        class_idx = pred_class if target_class is None else int(target_class)

        self.model.zero_grad(set_to_none=True)
        score = logits[0, class_idx]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations/gradients. Check target layer.")

        activations = self.activations
        gradients = self.gradients

        if activations.ndim == 4:
            # (B, C, H, W)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            cam = torch.nn.functional.interpolate(
                cam,
                size=(x.shape[-2], x.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            cam_map = cam[0, 0]
        elif activations.ndim == 3:
            # (B, C, L)
            weights = gradients.mean(dim=2, keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            cam = torch.nn.functional.interpolate(
                cam,
                size=(x.shape[-1],),
                mode="linear",
                align_corners=False,
            )
            cam_map = cam[0, 0]
        else:
            raise ValueError(
                f"Unsupported activation shape: {tuple(activations.shape)}. "
                "Use Conv1d/Conv2d/Conv3d-like layers."
            )

        cam_map = cam_map.detach().cpu()
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)

        return {
            "logits": logits.detach().cpu(),
            "pred_class": pred_class,
            "target_class": class_idx,
            "cam": cam_map,
        }


def build_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = str(Path(args.output_root) / args.model_type)
    return {
        "experiment_name": "grad_cam",
        "model_type": args.model_type,
        "annotation_dir": args.annotation_dir,
        "data_dir": "data/audio",
        "audio_dir": args.audio_dir,
        "output_dir": output_dir,
        "batch_size": 1,
        "epochs": 1,
        "lr": 1e-3,
        "seed": 42,
        "use_topk": False,
        "use_confusion": False,
        "num_classes": args.num_classes,
        "n_mels": args.n_mels,
        "max_frames": args.max_frames,
        "device": args.device,
    }


def save_visualization(
    input_tensor: torch.Tensor,
    cam: torch.Tensor,
    out_png: Path,
    title: str,
) -> bool:
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError:
        return False

    inp = input_tensor.detach().cpu()

    plt.figure(figsize=(10, 4))
    if inp.ndim == 3:
        # (1, H, W) -> spectrogram-like
        base = inp[0]
        plt.imshow(base, aspect="auto", origin="lower", cmap="gray")
        plt.imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.45)
        plt.colorbar(label="Grad-CAM")
        plt.xlabel("time")
        plt.ylabel("feature")
    elif inp.ndim == 2:
        # (1, T) waveform-like
        signal = inp[0].numpy()
        heat = cam.numpy()
        x = np.arange(signal.shape[0])
        plt.plot(x, signal, color="black", linewidth=1.0, label="signal")
        plt.plot(x, heat * np.max(np.abs(signal)), color="red", alpha=0.8, label="cam")
        plt.fill_between(
            x,
            signal.min(),
            signal.max(),
            where=heat > 0.3,
            color="red",
            alpha=0.2,
            interpolate=True,
            label="Grad-CAM",
        )
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.legend()
    else:
        raise ValueError(f"Unsupported input tensor shape for plotting: {tuple(inp.shape)}")

    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generalized Grad-CAM visualizer")
    parser.add_argument("--model-type", required=True, help="Registered model type (e.g. conformer, cnn2d, cnn1d)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--annotation-dir", default="data/annotation_data", help="Annotation JSON directory")
    parser.add_argument("--audio-dir", default="data/outputs/segment", help="Audio segment directory")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--target-class", type=int, default=None, help="Class index to explain (default: predicted class)")
    parser.add_argument("--layer", default=None, help="Target layer path (e.g. cnn_frontend.net.10)")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--output-root", default="results/cam", help="Root dir for CAM outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_plugins()

    cfg = build_runtime_config(args)
    strategy = create_model_strategy(cfg)
    strategy.build()

    if strategy.model is None:
        raise RuntimeError("Failed to build model from strategy.")

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_state = checkpoint.get("model_state_dict", checkpoint)
    strategy.model.load_state_dict(model_state, strict=True)
    strategy.model.to(args.device)
    strategy.model.eval()

    lazy_imports = strategy._lazy_imports()
    dataset_cls = (
        lazy_imports.dataset_cls
        if hasattr(lazy_imports, "dataset_cls")
        else lazy_imports[0]
    )
    dataset = dataset_cls(args.annotation_dir, args.audio_dir, config=cfg)

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample-index out of range: {args.sample_index} (dataset size={len(dataset)})"
        )

    x, y, metadata = dataset[args.sample_index]
    x_batched = x.unsqueeze(0).to(args.device)

    if args.layer:
        layer_name = args.layer
        target_layer = resolve_module(strategy.model, layer_name)
    elif hasattr(lazy_imports, "default_cam_layer") and lazy_imports.default_cam_layer:
        layer_name = lazy_imports.default_cam_layer
        target_layer = resolve_module(strategy.model, layer_name)
    else:
        try:
            layer_name, target_layer = find_last_conv_layer(strategy.model)
        except ValueError as exc:
            raise ValueError(
                f"{exc} model_type='{args.model_type}' may require a CAM method for non-conv models. "
                "Specify --layer if applicable."
            ) from exc

    cam_engine = GradCAM(strategy.model, target_layer)
    result = cam_engine(x_batched, target_class=args.target_class)
    cam_engine.remove_hooks()

    out_dir = Path(args.output_root) / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"sample_{args.sample_index}_class_{result['target_class']}"
    out_png = out_dir / f"{stem}.png"
    out_npz = out_dir / f"{stem}.npz"
    out_json = out_dir / f"{stem}.json"

    title = (
        f"model={args.model_type}, layer={layer_name}, "
        f"pred={result['pred_class']}, target={result['target_class']}, true={int(y)}"
    )
    image_saved = save_visualization(x, result["cam"], out_png, title)

    np.savez_compressed(
        out_npz,
        input=x.detach().cpu().numpy(),
        cam=result["cam"].numpy(),
        logits=result["logits"].numpy(),
        true_label=int(y),
        pred_class=int(result["pred_class"]),
        target_class=int(result["target_class"]),
    )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": args.model_type,
                "checkpoint": args.checkpoint,
                "layer": layer_name,
                "sample_index": args.sample_index,
                "true_label": int(y),
                "pred_class": int(result["pred_class"]),
                "target_class": int(result["target_class"]),
                "metadata": metadata,
                "output_png": str(out_png),
                "output_npz": str(out_npz),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if image_saved:
        print(f"Saved CAM image: {out_png}")
    else:
        print("matplotlib is not installed; skipped PNG output. Install matplotlib to render images.")
    print(f"Saved CAM arrays: {out_npz}")
    print(f"Saved metadata:  {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
