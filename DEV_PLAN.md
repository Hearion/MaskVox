# MaskVox 开发规划（交给 Codex 实现）

> 本文档是完整的实现说明。请严格按"里程碑"顺序推进，每个里程碑达到验收标准后再进入下一个。
> 文中标注 **【需核实】** 的地方是凭经验写的接口细节，实现前必须对照对应库/仓库的真实源码确认，不要照抄。

---

## 1. 项目概述

**一句话**：Windows 上的实时 AI 变声器。把麦克风声音实时转换成另一个音色，输出到虚拟麦克风，供微信 / QQ / Discord / KOOK / 游戏内语音使用。用途是朋友间娱乐整蛊。

**核心设计决定：主程序不依赖 PyTorch。**

项目分成两个部分：

| 部分 | 作用 | 依赖 | 体积目标 |
|---|---|---|---|
| **主程序** `app/` | 界面、音频、实时变声 | ONNX Runtime（DirectML）、numpy 等 | 运行环境 + 基础模型合计 300–600MB |
| **转换组件** `converter/` | 把社区的 `.pth` 音色转成 `.onnx`；导出基础模型 | PyTorch（**CPU 版**即可） | 约 1GB **【需核实】**，可选安装 |

主程序在任何情况下都**不得** `import torch`。主程序通过子进程调用转换组件。没装转换组件的用户（比如收到软件的朋友）仍然可以使用已经转换好的音色。

**第一版必须具备**

1. 实时变声：麦克风 → RVC（ONNX）推理 → 虚拟麦克风（VB-Cable），端到端延迟目标 ≤ 300ms。
2. 导入别人训练好的 RVC 音色：
   - `.pth`（+ 可选 `.index`，或包含它们的 `.zip`）：自动调用转换组件转成 ONNX。兼容 RVC v1 / v2、32k / 40k / 48k、带 f0 的音色；不带 f0 的音色见 5.2。
   - `.maskvox`（本项目的音色包，见第 9 节）：直接导入，不需要转换组件。
3. 导出音色包 `.maskvox`，方便发给没装转换组件的朋友。
4. 好看的图形界面（设计规范见第 7 节）。
5. 音色库管理：列表、切换、重命名、删除。
6. 参数调节：变调（半音）、检索比例、噪声门限、延迟档位、自己监听开关。
7. 一键安装脚本 + 一键启动脚本。

**第一版不做**（见第 10 节末尾）：模型训练、音效板、全局热键、DSP 轻量变声、自研虚拟声卡驱动。

**目标硬件**：Windows 10/11 x64，NVIDIA RTX 3070。因为用 DirectML，A 卡和 Intel 核显理论上也能跑，但只要求在 3070 上达到实时。CPU 模式只需"能跑不崩"。

---

## 2. 技术选型（已定，不要更换）

### 主程序

| 部分 | 选型 | 原因 |
|---|---|---|
| 语言 | Python 3.10（64 位） | 与转换组件共用语言；wheel 兼容性最好 |
| 推理 | `onnxruntime-directml` | 包体只有几十 MB，自带 DirectML，不需要安装 CUDA / cuDNN。**不要用 `onnxruntime-gpu`**，它依赖的 CUDA 运行库有 2GB 左右，会让体积目标落空 |
| 界面 | pywebview（WebView2）+ 原生 HTML/CSS/JS | 容易做好看；不起 HTTP 服务；无前端构建工具 |
| 音频 IO | sounddevice（PortAudio，WASAPI） | 延迟低 |
| 重采样 | `soxr`（python-soxr） | 体积小、质量高、支持流式 |
| 数值计算 | numpy | SOLA、梅尔谱、f0 解码全部用 numpy |
| 检索 | faiss-cpu | 读取 `.index` |
| 虚拟麦克风 | VB-Cable（用户自行安装） | 自研驱动不值得 |

### 转换组件

| 部分 | 选型 |
|---|---|
| PyTorch | 2.x **CPU 版** wheel（转换不需要 GPU，CPU 版比 CUDA 版小得多） |
| 模型定义 | 从 RVC 官方仓库拷贝（见 5.1） |
| ContentVec | transformers 加载 `lengyue233/content-vec-best` |
| 导出 | `torch.onnx.export`，opset 17；导出后用 `onnx` + `onnxruntime`（CPU）校验 |

**硬性约束**

- 主程序不得依赖 torch / torchaudio / transformers / librosa / scipy。加一条测试：在主程序的 venv 里 `import app.main` 后断言 `"torch" not in sys.modules`。
- 前端不使用任何框架和构建工具，只写原生 HTML + CSS + ES Module。
- 不起本地 HTTP 服务，UI 通过 `file://` 加载，前后端通过 pywebview 的 `js_api` 通信。
- 不注入任何进程，不 hook 任何游戏或聊天软件。
- 模型权重不入库（`.gitignore` 掉 `assets/`、`voices/`）。

---

## 3. 整体架构

```
┌──────────────┐   48k float32    ┌───────────────────────────────┐
│ 麦克风       │ ───────────────▶ │ 输入回调（sounddevice）       │
│ (WASAPI)     │                  │  只做：写入输入环形缓冲       │
└──────────────┘                  └──────────────┬────────────────┘
                                                 │
                                   ┌─────────────▼──────────────┐
                                   │ 推理线程（独占 ORT 会话）   │
                                   │ 取一个 block → 噪声门      │
                                   │ → 48k→16k → contentvec.onnx│
                                   │ → rmvpe.onnx → (检索)      │
                                   │ → 音色.onnx → SOLA 拼接    │
                                   │ → tgt_sr→48k               │
                                   └──────┬──────────────┬──────┘
                          ┌───────────────▼──┐   ┌───────▼───────────┐
                          │ 输出环形缓冲 A   │   │ 输出环形缓冲 B    │
                          └───────┬──────────┘   └───────┬───────────┘
                       ┌──────────▼─────────┐   ┌────────▼──────────┐
                       │ CABLE Input        │   │ 耳机（自己监听）  │
                       │ (VB-Cable)         │   │ 可选              │
                       └──────────┬─────────┘   └───────────────────┘
                       聊天/游戏软件把麦克风选为 "CABLE Output"

┌─────────────────────────────┐        子进程        ┌──────────────────────┐
│ 主程序（无 torch）          │ ───────────────────▶ │ 转换组件（有 torch） │
│  UI ⇄ js_api ⇄ 引擎/音色库  │ ◀── JSON 行进度 ──── │ .pth → .onnx         │
└─────────────────────────────┘                      └──────────────────────┘
```

**线程模型（重要）**

- 音频回调线程：只做内存拷贝，绝不做推理、绝不加会阻塞的锁、绝不分配大对象。
- 推理线程：一个，独占三个 ORT 会话。循环等待"输入缓冲里攒够一个 block"。
- UI：pywebview 主线程；`js_api` 的方法运行在 pywebview 的工作线程，对引擎的操作通过命令队列或加锁的参数对象传递。
- 参数热更新：变调、检索比例、噪声门限、监听音量可在运行中修改，推理线程每个 block 开头读取一次快照。切换音色、切换设备、改延迟档位需要"停止 → 重建 → 启动"。

---

## 4. 目录结构

```
maskvox/
├─ README.md
├─ DEV_PLAN.md
├─ requirements.txt              主程序依赖（无 torch）
├─ install.bat                   装主程序环境
├─ install_converter.bat         装转换组件环境 + 导出基础模型
├─ start.bat
├─ .gitignore
├─ app/
│  ├─ main.py                    入口
│  ├─ api.py                     暴露给 JS 的 Api 类
│  ├─ config.py                  路径常量、设置读写
│  ├─ events.py                  Python → JS 事件推送
│  ├─ audio/
│  │  ├─ devices.py              设备枚举、VB-Cable 检测
│  │  ├─ ringbuffer.py           线程安全的 numpy 环形缓冲
│  │  └─ stream.py               输入 / 输出 / 监听流
│  ├─ engine/
│  │  ├─ ort_session.py          创建 ORT 会话（provider 选择、选项）
│  │  ├─ content_encoder.py      contentvec.onnx 封装
│  │  ├─ f0_rmvpe.py             numpy 梅尔谱 + rmvpe.onnx + 解码 + coarse
│  │  ├─ index.py                faiss 检索
│  │  ├─ rvc_onnx.py             音色 onnx 封装
│  │  ├─ resample.py             soxr 封装
│  │  ├─ sola.py                 SOLA 对齐与交叉淡化（纯 numpy）
│  │  ├─ realtime.py             推理线程 + 分块
│  │  └─ passthrough.py          直通引擎（里程碑 1、调试用）
│  ├─ voices/
│  │  ├─ library.py              音色库增删改查
│  │  ├─ package.py              .maskvox 打包 / 解包 / 校验
│  │  └─ converter_client.py     以子进程调用转换组件
│  └─ ui/
│     ├─ index.html
│     ├─ css/ (tokens.css, layout.css, components.css)
│     └─ js/  (main.js, bridge.js, state.js, components/*.js)
├─ converter/
│  ├─ requirements.txt           torch(CPU)、transformers、onnx、onnxruntime 等
│  ├─ convert_voice.py           CLI：.pth → 音色 onnx + meta
│  ├─ export_base_models.py      CLI：导出 contentvec.onnx、rmvpe.onnx、rmvpe_mel.npz
│  ├─ verify.py                  导出结果与 torch 输出的数值比对
│  └─ vendor/                    RVC 官方模型定义（见 5.1）
├─ scripts/
│  ├─ offline_convert.py         wav 进 wav 出，走与实时完全相同的分块路径
│  └─ build_release.py           打出给朋友用的精简包（里程碑 6）
├─ tests/
├─ assets/                       基础模型（不入库）
└─ voices/                       用户音色库（不入库）
```

两个独立的虚拟环境：`.venv`（主程序）和 `converter/.venv`（转换组件）。路径统一从 `config.py` 取。

---

## 5. 模型部分

### 5.1 模型定义代码的来源（仅转换组件使用）

RVC 的网络定义不要自己重写（state_dict 的键名必须与社区 `.pth` 完全一致）。

- 来源：`https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI`（MIT 许可）。
- 需要：`infer/lib/infer_pack/` 下的 `models.py`、`models_onnx.py`、`attentions.py`、`commons.py`、`modules.py`、`transforms.py`；`infer/lib/rmvpe.py`；参考 `infer/modules/onnx/export.py`（官方 ONNX 导出脚本）。**【需核实：这些文件在所选 commit 中的实际路径】**
- 做法：**固定到一个具体 commit**，拷到 `converter/vendor/`，内部 import 改成相对导入，附原仓库 LICENSE（存为 `LICENSE_RVC`），在 `vendor/README.md` 写明来源 commit。
- 请通读该仓库的 `tools/rvc_for_realtime.py` 和 `gui_v1.py`，它们是实时管线的参考实现。本项目算法与之等价，但要用 numpy + ONNX Runtime 重新实现，不能直接搬（它们依赖 torch）。

### 5.2 音色转换：`.pth` → `.onnx`

**输入 `.pth` 的格式**：`torch.save` 的字典。

| 键 | 含义 |
|---|---|
| `weight` | 合成器 state_dict（不含 `enc_q`） |
| `config` | 构造参数列表，最后一项是目标采样率 |
| `f0` | 1 = 带音高，0 = 不带 |
| `version` | `"v1"` / `"v2"`，缺失按 `"v1"` |

**安全要求（必须做到）**：`.pth` 是 pickle，别人给的文件可能带恶意代码。

- 一律 `torch.load(path, map_location="cpu", weights_only=True)`。
- 加载失败时**不要**回退到 `weights_only=False`，直接报错："该音色文件格式不安全或已损坏，已拒绝加载"。
- 转换在独立子进程里进行，本身也是一层隔离。

**导出**（参照官方 `export.py`）**【需核实签名与输入名】**：

```python
cpt = torch.load(path, map_location="cpu", weights_only=True)
cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
version = cpt.get("version", "v1")
vec_dim = 256 if version == "v1" else 768
net_g = SynthesizerTrnMsNSFsidM(*cpt["config"], is_half=False, version=version)
net_g.load_state_dict(cpt["weight"], strict=False)
net_g.eval()

T = 200
dummy = (torch.rand(1, T, vec_dim), torch.tensor([T]).long(),
         torch.randint(5, 255, (1, T)), torch.rand(1, T),
         torch.LongTensor([0]), torch.rand(1, 192, T))
torch.onnx.export(net_g, dummy, out_path, opset_version=17, do_constant_folding=False,
    input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
    output_names=["audio"],
    dynamic_axes={"phone": [1], "pitch": [1], "pitchf": [1], "rnd": [2]})
```

- `rnd` 是外部传入的随机噪声，形状 `[1, 192, T]`，主程序每次用 numpy 生成标准正态噪声。
- 元数据写进 ONNX 的 `metadata_props`：`mv_format=1`、`version`、`sr`、`f0`、`vec_dim`、`source_commit`。主程序只认带 `mv_format` 的 onnx。
- **不带 f0 的音色**：官方 ONNX 导出只覆盖带 f0 的模型。第一版对 `f0 == 0` 的音色给出明确提示"暂不支持不带音高的音色"，记入后续事项。社区音色绝大多数带 f0。
- **导出后必须校验**（`verify.py`）：固定同一组输入和同一份 `rnd`，比较 torch 输出与 onnxruntime(CPU) 输出，要求最大绝对误差 < 1e-3 **【阈值需按实际情况调整并记录】**；再用两个不同的 `T` 验证动态长度可用。校验失败则转换失败，不产出文件。

**CLI 约定**：

```
converter\.venv\Scripts\python -m converter.convert_voice --pth <路径> --out <目录>
```

- 标准输出每行一个 JSON：`{"stage": "loading" | "exporting" | "verifying" | "done", "progress": 0~1}`；失败时 `{"stage": "error", "message": "中文原因"}` 并以非零码退出。
- 输出目录里产出 `model.onnx` 和 `meta.json`。
- 主程序侧设置超时（如 5 分钟），子进程的 stderr 写入日志。

**性能优化项（M4 之后再做，先保证正确）**：官方导出每次都解码整段上下文。可以自己写一个导出包装类：`enc_p` 处理全长特征，然后只取尾部若干帧过 `flow` 和 `dec`，裁剪起点作为 int64 输入。等价于 PyTorch 版 `infer` 的 `skip_head` / `return_length`。做了这个优化后同样要过 `verify.py`。

### 5.3 基础模型导出（`export_base_models.py`）

产出三个文件放到 `assets/`：

**`contentvec.onnx`**
- 用 transformers 加载 `lengyue233/content-vec-best`。**【需核实：按该模型卡片定义带 `final_proj` 的 `HubertModel` 子类】**
- 包一层导出：输入 `audio`（`[1, T]`，16k float32），两个输出：
  - `feats768`：最后一层输出，给 v2 音色。
  - `feats256`：第 9 层隐藏状态过 `final_proj`，给 v1 音色。
- `T` 为动态轴。输出帧率 50fps。
- 用 verify 比对 torch 与 ORT 输出。

**`rmvpe.onnx` + `rmvpe_mel.npz`**
- 权重 `rmvpe.pt` 来自 Hugging Face `lj1995/VoiceConversionWebUI`。该仓库可能直接提供 `rmvpe.onnx`，可以直接用 **【需核实是否存在、输入输出定义是什么】**；否则用 vendor 的 `rmvpe.py` 自行导出。
- ONNX 的输入是梅尔谱 `[1, 128, n_frames]`，输出是音高显著图 `[1, n_frames, 360]`。梅尔谱不放进图里，由主程序用 numpy 计算。
- `rmvpe_mel.npz` 保存主程序算梅尔谱需要的常量：梅尔滤波器组矩阵、窗函数，以及参数（`sr=16000, n_fft=1024, hop=160, n_mels=128, fmin=30, fmax=8000`）。从 vendor 的 `MelSpectrogram` 实例里直接取出来存，保证与训练时一致。**【需核实参数】**

**FP16**：第一版全部用 FP32。M5 再尝试把基础模型转 FP16（`onnxconverter-common`，保留 FP32 输入输出）以减半体积，转完必须重新校验且实测 DirectML 下无异常。

**中国大陆网络**：Hugging Face 默认走 `HF_ENDPOINT=https://hf-mirror.com`，pip 默认走清华源，均可用环境变量覆盖。下载失败时给出手动下载地址和应放置的路径。

### 5.4 主程序侧：ORT 会话（`ort_session.py`）

```python
so = ort.SessionOptions()
so.enable_mem_pattern = False                 # DirectML 要求
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
providers = [("DmlExecutionProvider", {"device_id": gpu_index}), "CPUExecutionProvider"]
```
**【需核实：DirectML 的必要会话选项与 provider 参数】**

- `ort.get_available_providers()` 里没有 Dml 时回退 CPU，并在界面提示"未启用显卡加速"。
- 设置里提供"显卡选择"（device_id），应对双显卡笔记本选到核显的情况。
- 非 Windows 平台（开发、跑测试）安装普通 `onnxruntime`，用 CPU provider。`requirements.txt` 用平台标记区分两个包。
- 实时运行时每次调用的输入长度是固定的（由延迟档位决定），这对 DirectML 的性能很重要：同一档位下不要让输入形状变化。
- 加载音色后先用静音跑 2~3 次完整推理做预热，再开始出声。

### 5.5 主程序侧：内容特征

- 输入 16k 缓冲，取 `feats768` 或 `feats256`（由音色 meta 的 `vec_dim` 决定）。
- 时间轴 2 倍插值到 100fps（`np.repeat(feats, 2, axis=1)`，与参考实现的最近邻插值等价）。

### 5.6 主程序侧：音高（`f0_rmvpe.py`，纯 numpy）

1. 梅尔谱：反射填充 → 分帧加窗 → `np.fft.rfft` → 取模 → 乘梅尔滤波器组 → `log(clip(x, 1e-5))`。帧数补齐到 32 的倍数再送入 ONNX，输出再裁回原帧数。**【对照 vendor `rmvpe.py` 的 `mel2hidden` 核实填充方式】**
2. 解码：对每帧 360 维显著图，取 argmax 附近 ±4 个 bin 做加权平均得到 cents（`cents_mapping = 20 * arange(360) + 1997.3794084376191`），`f0 = 10 * 2 ** (cents / 1200)`；最大显著度低于阈值 0.03 的帧置 0。**【对照 vendor 的 `decode` / `to_local_average_cents` 核实】**
3. 变调：`f0 *= 2 ** (semitones / 12)`。
4. f0 → coarse：

   ```python
   f0_min, f0_max = 50, 1100
   mel_min = 1127 * log(1 + f0_min / 700);  mel_max = 1127 * log(1 + f0_max / 700)
   f0_mel = 1127 * log(1 + f0 / 700)
   f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - mel_min) * 254 / (mel_max - mel_min) + 1
   f0_mel[f0_mel <= 1] = 1;  f0_mel[f0_mel > 255] = 255
   coarse = rint(f0_mel).astype(int64)
   ```
5. 实时优化：只对"本次新增音频 + 少量上下文"算 f0，结果写入滚动缓存，不要每个 block 对整段上下文重算。
6. 要有一个单元测试：把 numpy 梅尔谱与 vendor torch 版梅尔谱在同一段音频上比对（该测试在转换组件环境里运行）。

### 5.7 主程序侧：检索（`.index`）

- 可选。有 index 且检索比例 > 0 时启用。
- `index = faiss.read_index(path)`；`big_npy = index.reconstruct_n(0, index.ntotal)`。
- 每帧取 top-8：`score, ix = index.search(feats, 8)`；`w = 1 / score**2` 归一化；`retrieved = sum(big_npy[ix] * w[..., None], axis=1)`；`feats = retrieved * rate + feats * (1 - rate)`。注意处理 `score == 0`。
- 在 2 倍插值**之前**做；只对需要输出的那部分帧做。
- index 维度必须等于音色的 `vec_dim`，不匹配则忽略并提示。

### 5.8 主程序侧：合成

```python
audio = sess.run(["audio"], {
    "phone": feats,                       # [1, T, vec_dim] float32
    "phone_lengths": np.array([T], np.int64),
    "pitch": coarse[None],                # [1, T] int64
    "pitchf": f0[None].astype(np.float32),
    "ds": np.array([0], np.int64),
    "rnd": np.random.randn(1, 192, T).astype(np.float32),
})[0]
```
输出是 `tgt_sr` 采样率的波形，取尾部需要的长度。**【需核实输出形状与每帧对应的采样点数：`tgt_sr / 100`】**

---

## 6. 实时管线

### 6.1 采样率约定

- 声卡流统一 **48000 Hz、单声道、float32**。打开 WASAPI 流时使用 `sd.WasapiSettings(auto_convert=True)`。**【需核实当前 sounddevice 是否支持；不支持则按设备默认采样率开流并内部重采样】**
- 内部：48k → 16k 给内容编码器和 f0；合成器输出 `tgt_sr` → 48k。
- 用 soxr 的流式重采样器，实例常驻，不要每个 block 新建。

### 6.2 分块参数

以 48k 为例，`zc = 480`（10ms，所有长度对齐到 zc 的整数倍）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `block_time` | 0.20s | 三档：低延迟 0.12 / 均衡 0.20 / 高质量 0.35 |
| `extra_time` | 1.0s | 历史上下文。官方 ONNX 导出每次解码整段上下文，所以先取 1.0s；做了 5.2 的裁剪优化后可加到 2.0s |
| `crossfade_time` | 0.05s | 块间交叉淡化 |
| `sola_search` | 0.01s | SOLA 搜索范围 |

缓冲区长度：`extra + crossfade + sola_search + block`。

### 6.3 每个 block 的处理步骤

1. 从输入环形缓冲取 `block_frame` 个采样点。
2. 噪声门：按 10ms 小段算 RMS（dB），低于门限的清零。−60dB 表示关闭。
3. 滚动输入缓冲，同步维护 16k 版本（只对新增部分重采样）。
4. 推理：16k 缓冲 → 内容特征 → 检索 → 插值 → f0（滚动缓存）→ 合成，取尾部 `block + crossfade + sola_search` 对应的输出。
5. 输出重采样到 48k。
6. **SOLA 对齐**（numpy）：在输出开头 `sola_buffer + sola_search` 范围内与上一块保留的 `sola_buffer` 做归一化互相关：

   ```python
   seg = infer_wav[: sola_buffer_frame + sola_search_frame]
   nom = np.correlate(seg, sola_buffer, mode="valid")
   den = np.sqrt(np.convolve(seg ** 2, np.ones(sola_buffer_frame), mode="valid") + 1e-8)
   offset = int(np.argmax(nom / den))
   infer_wav = infer_wav[offset:]
   ```
7. 交叉淡化：`infer_wav[:n] = infer_wav[:n] * fade_in + sola_buffer * fade_out`，窗函数 `sin²` / `cos²`。
8. 保存新的 `sola_buffer = infer_wav[block_frame : block_frame + sola_buffer_frame]`。
9. `infer_wav[:block_frame]` 写入输出缓冲 A；开启监听时乘以监听音量写入缓冲 B。
10. 统计推理耗时、输入/输出电平，每 100ms 推送给界面一次。

`sola_buffer_frame = min(crossfade_frame, 4 * zc)`，与参考实现一致。

### 6.4 欠载与溢出

- 输出回调取不到足够数据时补零并计数。
- 输出缓冲积压超过 `2 × block` 时丢弃最旧数据，避免延迟越积越大。
- 监听设备与虚拟声卡是两个独立时钟，各用独立的环形缓冲。
- 推理耗时持续大于 `block_time` 时界面显示"性能不足"，建议切到更高延迟档位。

### 6.5 延迟显示

`总延迟 ≈ block_time + 推理耗时 + 设备缓冲`。状态栏实时显示"推理 xx ms / 总延迟约 xxx ms"。

### 6.6 设备

- 只列 WASAPI 主机 API 下的设备，避免同一设备重复出现。
- 自动检测 VB-Cable：输出设备名含 `CABLE Input`。找到则默认选中；找不到则顶部显示引导条和"打开下载页"按钮（`https://vb-audio.com/Cable/`）。
- 输入设备列表排除 `CABLE Output`，防止回环。
- 热插拔：第一版只提供"刷新设备"按钮。

---

## 7. 界面设计规范

### 7.1 总体

- 窗口 1080 × 700，最小 960 × 640，可缩放。标题 "MaskVox"。
- 深色主题，单页面。所有文案用中文。
- 风格：干净、有质感，像一台"设备"而不是一张表单。大面积深色底 + 一个强调色 + 克制的发光。
- 不使用外部 CDN、字体、图标库（要能离线运行）。图标用内联 SVG。字体栈：`"Segoe UI Variable", "Segoe UI", "Microsoft YaHei UI", sans-serif`。

### 7.2 设计变量（`tokens.css` 的 `:root`）

```css
--bg-0: #0b0d12;   --bg-1: #12151c;   --bg-2: #1a1e28;   --bg-3: #242a37;
--line: rgba(255,255,255,.07);
--text-0: #eef1f6; --text-1: #a4acbb; --text-2: #6b7385;
--accent: #7c5cff;        /* 主强调色：紫 */
--accent-2: #22d3ee;      /* 辅助：青，用于渐变和电平 */
--ok: #34d399;  --warn: #fbbf24;  --danger: #f87171;
--radius-s: 8px;  --radius-m: 14px;  --radius-l: 22px;
--shadow-glow: 0 0 40px rgba(124,92,255,.35);
--ease: cubic-bezier(.2,.8,.2,1);
```

### 7.3 布局

```
┌────────────────────────────────────────────────────────────────────┐
│ ◉ MaskVox                     [未检测到虚拟麦克风 · 去安装]  ⚙  │ 顶栏 52px
├───────────────┬───────────────────────────────┬────────────────────┤
│ 音色库        │                               │ 调节               │
│ [+ 导入音色]  │        ╭───────────╮          │ 变调      ──●──  +3│
│               │        │   电源    │          │ 音色贴合  ───●─ 0.5│
│ ▣ 音色卡片 A  │        │   大按钮  │          │ 噪声门    ─●─── -45│
│ ▣ 音色卡片 B  │        ╰───────────╯          │ 延迟档位 [低|均衡|高]│
│ ▣ 音色卡片 C  │   当前音色：xxx  v2 · 40k     │ ───────────────    │
│   …           │   ▁▃▅▇▅▃▁  实时波形/电平      │ 麦克风  [下拉]     │
│               │   输入 ▮▮▮▮▮▯▯   输出 ▮▮▮▯▯▯  │ 输出到  [下拉]     │
│               │                               │ 自己监听 (开关)    │
│               │                               │ 监听设备 [下拉]    │
│  280px        │            自适应             │        300px       │
├───────────────┴───────────────────────────────┴────────────────────┤
│ ● 运行中   推理 38ms · 总延迟约 250ms          显卡加速：DirectML  │ 状态栏 36px
└────────────────────────────────────────────────────────────────────┘
```

右上角 ⚙ 打开设置抽屉：显卡选择、转换组件状态（已安装 / 未安装 + 安装说明）、日志目录、关于。

### 7.4 组件细节

**电源大按钮（视觉中心）**
- 直径 180px 的圆。关闭：`--bg-2` 底、细描边、灰色电源图标。
- 开启：紫→青渐变描边、`--shadow-glow`、外圈光环随输出电平缩放（JS 每帧更新 CSS 变量，用 `transform` + `opacity`）。
- 启动中：描边转圈，按钮禁用，下方文字"正在加载音色…"。
- 下方显示当前音色名和 `v2 · 40k` 小标签。

**实时波形**
- `<canvas>` 高 96px，画最近约 2 秒的输出电平条（圆角竖条、上下镜像），accent 渐变。未运行时是一条静止细线。
- `requestAnimationFrame` 驱动，数据来自后端每 100ms 的电平推送，前端插值平滑。

**音色卡片**
- 高 64px。左侧 40px 圆角方块头像：按名字哈希生成双色渐变，中间显示首字。
- 名称（单行省略）+ 标签行（`v2`、`40k`、有无 `index`）。
- 选中态：左侧 3px accent 竖条 + `--bg-3`。
- 悬停出现"⋯"菜单：重命名 / 导出音色包 / 删除（删除二次确认）。
- 空状态：虚线框 + "把 .pth / .zip / .maskvox 拖到这里，或点击导入"。

**导入**
- "导入音色"→ 后端打开系统文件对话框（多选，过滤 `.pth .index .zip .maskvox`）。
- 拖拽导入：整个左栏是放置区。pywebview 中拿到真实文件路径的方式 **【需核实：pywebview 5.x 的 DOM 事件 + `pywebviewFullPath`】**；做不通就只保留按钮导入。
- 导入 `.pth` 时卡片位置显示转换进度条和阶段文字（读取 / 转换 / 校验），可取消。转换通常需要几十秒。
- 导入 `.pth` 但未安装转换组件：弹对话框说明"导入 .pth 需要先安装转换组件（约 1GB）"，给出"查看安装方法"按钮；同时说明 `.maskvox` 音色包不需要它。
- 成功后新卡片带入场动画并自动选中；失败用 toast 显示后端返回的中文原因。

**滑块**
- 自绘：4px 轨道、已填充部分 accent、16px 圆形滑块，拖动时放大并显示数值气泡。双击恢复默认。
- 变调：−12 ~ +12，步长 1，默认 0。快捷按钮："男→女 +12"、"女→男 −12"。
- 音色贴合（检索比例）：0 ~ 1，步长 0.05，默认 0.5。无 index 时禁用并提示。
- 噪声门：−60 ~ −20 dB，默认 −45。

**分段选择器**：延迟档位。运行中修改会重启引擎，显示"正在切换…"。

**Toast**：右下角堆叠，3 秒消失，错误类需手动关闭。

**顶部引导条**：未检测到 VB-Cable 时显示（warn 色）。

**首次使用提示**：三步卡片：① 安装 VB-Cable ② 导入一个音色 ③ 在聊天软件里把麦克风选成 "CABLE Output"。可关闭并记住。

### 7.5 动效

过渡 160~240ms，使用 `--ease`，尊重 `prefers-reduced-motion`。只有电源按钮和波形是持续动画。

---

## 8. 前后端接口

### 8.1 JS → Python（`window.pywebview.api.*`，返回 Promise）

统一返回 `{ ok: true, data }` 或 `{ ok: false, error: "中文错误信息" }`。后端方法全部 try/except。

| 方法 | 说明 |
|---|---|
| `get_state()` | 设置、音色列表、设备列表、引擎状态、推理后端、是否检测到 VB-Cable、基础模型是否齐全、转换组件是否已安装 |
| `list_devices()` | |
| `set_devices({input_id, output_id, monitor_id})` | |
| `pick_and_import_voices()` | 弹文件对话框并导入 |
| `import_voice_paths(paths)` | 拖拽导入 |
| `cancel_import(job_id)` | 终止转换子进程 |
| `list_voices()` / `rename_voice(id, name)` / `delete_voice(id)` | |
| `export_voice(id)` | 弹保存对话框，导出 `.maskvox` |
| `select_voice(id)` | 运行中则热切换 |
| `start()` / `stop()` | |
| `set_param(name, value)` | `pitch` / `index_rate` / `gate_db` / `monitor` / `monitor_volume` / `latency_mode` / `gpu_index` |
| `open_url(url)` | 仅白名单地址 |
| `dismiss_onboarding()` | |

### 8.2 Python → JS（`window.evaluate_js("window.onEngineEvent(" + json + ")")`）

| 事件 | 载荷 |
|---|---|
| `status` | `{ state: "idle" \| "loading" \| "running" \| "error", message }` |
| `meter` | `{ in_db, out_db, infer_ms, latency_ms, underruns }`，每 100ms |
| `warning` | `{ code, message }` |
| `import_progress` | `{ job_id, file, stage, progress }` |
| `voices_changed` | 音色库有变化，前端重新拉列表 |

窗口加载完成后才能调 `evaluate_js`；关闭窗口时先停引擎、终止转换子进程，再退出。

### 8.3 前端状态

`state.js` 维护单一状态对象 + 订阅函数，不引入任何库。`bridge.js` 在没有 `window.pywebview` 时使用内置 mock（假设备、假音色、假电平、假转换进度），这样用浏览器直接打开 `index.html` 就能调界面。

---

## 9. 音色库与音色包

**库目录**：`voices/<id>/`，含 `model.onnx`、可选 `model.index`、`meta.json`。`id` 为 uuid4 前 8 位。

**`meta.json`**：`{ id, name, version, sr, f0, vec_dim, has_index, imported_at, source_filename }`。

**导入 `.pth` / `.zip`**

1. `.zip`：解压到临时目录，只取 `.pth` 和 `.index`。**必须防路径穿越**，并限制解压后总大小（2GB）和文件数。
2. 多个 `.pth` 时每个是一个音色；`.index` 按文件名相似度配对，配不上就不带。有多个 `.index` 时优先文件名以 `added_` 开头的。
3. 调转换组件（5.2）。训练检查点（`G_xxx.pth` / `D_xxx.pth`，没有 `weight` 键）提示："这是训练检查点，不是可用于推理的音色文件"。
4. 转换成功后把 `model.onnx`、index、meta 存入库。默认名取文件名，重名加序号。

**音色包 `.maskvox`**

- 本质是 zip，内含 `model.onnx`、可选 `model.index`、`meta.json`，不压缩或低压缩即可（权重压不动）。
- 导入时校验：防路径穿越；只接受这三个固定文件名；`model.onnx` 必须带 `mv_format` 元数据；**拒绝引用外部数据文件的 onnx**（检查 initializer 的 `data_location`），防止读取包外的任意文件。
- 导入 `.maskvox` 不需要转换组件。

**删除**：删除整个 `voices/<id>/`；若正在使用先停引擎。

---

## 10. 里程碑与验收标准

**每完成一个里程碑：跑通测试，更新 README，再继续。**

### M0 项目骨架
- 目录、两份 `requirements.txt`、`.gitignore`、三个 bat、空壳 `main.py`。
- `install.bat`：检查 Python 3.10 → 建 `.venv` → 装主程序依赖。全程不出现 torch。
- `install_converter.bat`：建 `converter/.venv` → 从 PyTorch 官方 CPU 源装 torch → 装其余依赖 → 运行 `export_base_models.py` 生成 `assets/`。
- `start.bat`：激活 `.venv`，运行 `python -m app.main`，出错时 `pause`。
- 所有 bat：开头 `chcp 65001`，路径加引号，每步失败有中文提示并停下。
- 验收：`start.bat` 打开显示 "MaskVox" 的空窗口；主程序 venv 里 `pip list` 没有 torch；`python -c "import onnxruntime as o; print(o.get_available_providers())"` 包含 `DmlExecutionProvider`。

### M1 音频直通
- 完成 `audio/`、`passthrough.py`。
- 验收：选麦克风 + CABLE Input，另一软件把麦克风选成 CABLE Output 能听到原声；连续 10 分钟延迟不增长；环形缓冲测试通过。

### M2 界面（接 mock 和直通）
- 完成第 7、8 节，引擎先接直通，导入先接假进度。
- 验收：浏览器直接打开 `index.html`（mock 模式）所有交互可用；pywebview 中设备选择、启停、电平、波形正常；最小窗口尺寸下布局不乱。

### M3 转换组件
- 完成 `converter/`：vendor、`export_base_models.py`、`convert_voice.py`、`verify.py`。
- 验收：`assets/` 下生成三个基础模型文件且通过数值校验；至少 2 个真实社区音色（一个 v1、一个 v2）转换成功并通过校验；损坏文件、训练检查点、不带 f0 的音色都给出对应的中文错误；numpy 梅尔谱与 torch 梅尔谱比对测试通过。

### M4 离线推理 + 实时变声
- 完成 `engine/` 其余部分、`voices/`、`scripts/offline_convert.py`，接入界面。
- `offline_convert.py --voice <id> --in a.wav --out b.wav --pitch 0`，内部走和实时**完全相同**的分块 + SOLA 路径。
- 验收：
  - 离线：30 秒语音转换结果听感正常、块边界无爆音。
  - 对照：同一音色、同一输入，本项目离线输出与官方 RVC（PyTorch）的转换结果听感一致（允许细微差异，不允许音高错误、明显失真或音色不对）。
  - 实时（RTX 3070，均衡档）：推理耗时 < `block_time` 的 60%；连续 30 分钟无欠载累积、内存不增长；拖动变调即时生效；运行中切换音色数秒内完成；Discord 或微信语音里对方听到变声后的声音。
  - 记录三档延迟下的实测推理耗时，写进 README。如果均衡档达不到要求，先做 5.2 末尾的裁剪优化。

### M5 打磨
- 错误处理全覆盖（设备被拔、显存不足、基础模型缺失、音色文件被外部删除、转换子进程崩溃），都有中文提示且程序不退出。
- 设置持久化，下次启动恢复但不自动开始变声。
- 尝试 FP16 基础模型（见 5.3），通过校验和实测才启用。
- README：安装、VB-Cable 设置、各聊天软件怎么选麦克风、常见问题（没声音 / 回声 / 延迟大 / 游戏掉帧）。

### M6 精简分发包
- `scripts/build_release.py` 产出一个文件夹（或 zip）：内嵌 Python 运行环境 + 主程序依赖 + `app/` + `assets/` + 启动器，**不含**转换组件。朋友解压后双击即可运行，不需要自己装 Python。
- 可用 PyInstaller（onedir）或 python-embed 方案，选更稳的一种。
- 验收：在一台没装 Python 的 Windows 机器上解压即可运行；导入 `.maskvox` 可用；记录发布包的实际体积，目标 ≤ 600MB（FP16 后争取 ≤ 400MB）；包内搜不到 torch。

### 后续（不在本次范围）
不带 f0 的音色、FCPE 音高、音效板、全局热键、DSP 轻量变声、RNNoise 降噪、应用内一键安装转换组件、安装程序。

---

## 11. 测试策略

- **不依赖 GPU 和真实模型的测试**：环形缓冲、SOLA（正弦波拼接后相位连续、无跳变）、f0 → coarse、音色包校验（含路径穿越、外部数据引用）、主程序不导入 torch。
- **假模型管线测试**：用 `onnx.helper` 手工构造几个极小的假 onnx（输入输出名和形状与真模型一致，内部随便算），跑通整条分块管线，验证输出长度、无 NaN、块间连续。不需要 torch。
- **转换组件测试**（在 `converter/.venv` 里跑）：用 vendor 的类按合法 config 随机初始化一个假 `.pth`，走完整转换 + 校验流程。
- **真机验证**（GPU 性能、WASAPI、VB-Cable、聊天软件实测）只能在目标 Windows 机器上做。如果开发环境不是这台机器，请完成所有可离线验证的部分，并在 README 的"待真机验证"清单里逐条列出没验证的内容，不要声称已验证。

---

## 12. 已知的坑

1. **回声**：监听默认关闭，开启时提示"请佩戴耳机"。
2. **weight_norm**：vendor 代码使用旧的 `torch.nn.utils.weight_norm`（键为 `weight_g` / `weight_v`），不要换成 parametrizations 版本。导出前按官方做法处理（如 `remove_weight_norm`）。**【需核实官方导出流程是否调用】**
3. **ONNX 导出的随机性**：合成器里的噪声由 `rnd` 输入提供；正弦激励源内部若还有随机数，会导致 torch 与 ORT 输出无法逐点比对，校验时需要固定种子或改用统计/频谱指标。**【需核实 `models_onnx.py` 的实现】**
4. **DirectML 与动态形状**：形状频繁变化会很慢。同一延迟档位下保持所有输入长度恒定。
5. **DirectML 算子支持**：个别算子可能回退到 CPU，导致性能骤降。M4 时打开 ORT 的详细日志确认没有大面积回退。
6. **Hugging Face 在国内不可直连**：默认走镜像，失败给手动下载指引。
7. **WASAPI 采样率不匹配 / 设备被独占**：打开流失败时把 PortAudio 错误原文和中文解释一起显示。
8. **pywebview 的 `js_api`**：暴露的类里不要放非下划线开头的大对象属性，引擎等对象用下划线前缀。**【需核实】**
9. **GIL 与音频回调**：推理线程里 ORT 的 `run` 会释放 GIL，但 numpy 的梅尔谱和 SOLA 计算会占用。输出侧预填 1 个 block 的静音，回调里只做切片拷贝。
10. **游戏占用 GPU**：游戏满载时推理耗时会上升，6.4 的性能警告为此准备。
11. **中文路径**：所有路径用 `pathlib`。faiss 和 onnxruntime 读取含中文的路径可能失败：音色库内部文件名固定为 ASCII；库目录本身含中文仍失败时，index 先读成字节再 `faiss.deserialize_index`，onnx 先读成字节再传给 `InferenceSession`。

---

## 13. 代码规范

- Python：类型标注、`pathlib`、`logging`（写到 `logs/maskvox.log`，滚动保存）。
- 面向用户的文字（界面、错误、bat 提示、README）用简体中文；代码标识符和注释用英文。
- 任何异常都不能让程序静默退出或卡死：捕获 → 记日志 → 通过事件告知界面。
- 每个里程碑至少一次提交，提交信息写清楚做了什么。

## 14. 使用边界

本软件用于朋友间娱乐。README 需写明：请勿用于冒充他人、诈骗或其他违法用途；导入的音色请确保来源合法。软件本身不内置任何真人音色。
