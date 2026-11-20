"""Confirm that an exported graph agrees with its PyTorch reference on CPU."""

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


def export_graph(
    model: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    path: Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: dict[str, dict[int, str]],
) -> None:
    with torch.inference_mode():
        torch.onnx.export(
            model,
            args,
            str(path),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
            external_data=False,
            do_constant_folding=True,
        )


def verify_export(
    model: torch.nn.Module,
    path: Path,
    input_sets: Iterable[tuple[torch.Tensor, ...]],
    output_names: Sequence[str],
    *,
    max_abs: float,
    expected_shapes: Sequence[Sequence[tuple[int, ...]]] | None = None,
) -> float:
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    inputs = [node.name for node in session.get_inputs()]
    largest_error = 0.0
    with torch.inference_mode():
        for case, args in enumerate(input_sets):
            reference = model(*args)
            if isinstance(reference, torch.Tensor):
                reference = (reference,)
            feeds = {name: arg.numpy() for name, arg in zip(inputs, args, strict=True)}
            results = session.run(list(output_names), feeds)
            for i, (output, wanted) in enumerate(zip(results, reference, strict=True)):
                if output.size == 0 or output.shape != tuple(wanted.shape):
                    raise ValueError(f"ONNX 输出 {output_names[i]} 的形状与 PyTorch 不一致")
                if expected_shapes is not None and output.shape != expected_shapes[case][i]:
                    raise ValueError(f"ONNX 输出 {output_names[i]} 的帧长不正确")
                if not np.isfinite(output).all() or not torch.isfinite(wanted).all():
                    raise ValueError(f"ONNX 校验发现 {output_names[i]} 输出非有限数值")
                error = float(np.max(np.abs(output - wanted.numpy())))
                if error >= max_abs:
                    raise ValueError(
                        f"ONNX 输出 {output_names[i]} 的最大误差 {error:.6g} 超过限值 {max_abs:g}"
                    )
                largest_error = max(largest_error, error)
    return largest_error
