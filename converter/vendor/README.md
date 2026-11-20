# RVC 上游模型实现

代码来自 Retrieval-based-Voice-Conversion-WebUI 的 `20240604` 分支，固定提交 `d60cd3d009b7e438f92f8624fa89b710d10c6a39`，依据同目录的 `LICENSE_RVC` 分发。模型代码来自上游 `infer/lib/infer_pack/` 及 `infer/lib/rmvpe.py`，ONNX 导出依照上游 `infer/modules/onnx/export.py`。实时对照使用同一提交中的 `tools/rvc_for_realtime.py` 和 `gui_v1.py`。

- 合成器使用包内相对导入，沿用 `nn.utils.weight_norm` 的权重名。导出前折叠权重归一化。
- ONNX 合成器允许从额外的 `source_noise` 输入传入样本级标准正态噪声。相位、浊音掩码与噪声幅度由原有正弦激励模块计算。这样验证时能将同一份噪声同时交给 PyTorch 和 ONNX Runtime。
- 注意力层同时支持 ONNX 跟踪传入的张量长度与 PyTorch 验证使用的整数长度。
- RMVPE 使用 CPU 导出所需的 `E2E` 和 `MelSpectrogram`，移除了依赖原仓库的 JIT 与 Intel 扩展引导。

转换组件支持使用标准 RVC v1 和 v2 架构训练并导出的带音高权重。早期实验权重如 `白菜357k.pt` 使用不同的说话人条件，导入时会提示权重不匹配。

[固定的上游提交](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/d60cd3d009b7e438f92f8624fa89b710d10c6a39)
