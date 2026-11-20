# MaskVox

MaskVox 是 Windows 10/11 x64 上的实时 RVC 变声器。它把麦克风输入转换为所选音色，可输出到 VB-Cable，供语音聊天和游戏使用。变声在本机运行。

## 开始使用

1. 将便携版完整解压到可写目录，双击其中的 `start.bat`。便携版已包含 Python 和基础模型，可直接导入 `.maskvox`。
2. 点击「导入音色」或把音色包拖入音色库。通过卡片右侧的菜单可重命名、导出或删除。
3. 选好麦克风和输出设备，点击中间的电源按钮。只想在本机试听时，把主输出选为耳机。
4. 要给微信、QQ、Discord、KOOK 或游戏使用，请安装 VB-Cable。解压驱动包，以管理员身份运行其中的 x64 安装程序并重启 Windows。 [VB-Cable 官方下载](https://vb-audio.com/Cable/)
5. 回到 MaskVox，刷新音频设备，把「输出到」设为 **CABLE Input**。在聊天或游戏软件的语音设置里，将麦克风设为 **CABLE Output**，扬声器仍选日常使用的耳机或音箱。

在 Discord 的「语音和视频」、KOOK 的语音设置，以及微信和 QQ 通话或应用的音频设置里查找麦克风选项。没有单独选项的应用可以跟随 Windows「设置 → 系统 → 声音 → 输入」里的默认设备。需要退出原有通话或重启应用，设备列表才可能刷新。

软件用于朋友间的娱乐。请勿用于冒充他人、诈骗或其他违法用途。请使用你有权使用的音色，并确认音色包和索引的来源。发行包里的基础模型用于提取内容特征和音高。变声目标音色由你导入。

## 导入社区 RVC 音色

源码版支持带音高的 RVC v1 和 v2 音色，采样率为 32 kHz、40 kHz 或 48 kHz。可以导入单个 `.pth`、一并选择配套的 `.index`，或导入包含音色和索引的 `.zip`。导入时会转换成 ONNX，并检查两种输入长度的推理结果。转换期间可取消，已完成的音色会保留在库中。

- 在源码目录双击 `install.bat` 安装主程序。脚本使用 Python 3.10 x64，必要时把 Python 官方的 NuGet 运行环境解压到项目内。
- 双击 `install_converter.bat` 安装转换组件并下载、校验、导出基础模型。转换组件的独立环境约 1.16 GB，下载缓存和 FP32 基础模型另外占用约 1.3 GB。
- 双击源码目录的 `start.bat`，导入社区音色。要分享给便携版用户，在音色菜单中选择「导出音色包」并保存为 `.maskvox`。

安装脚本默认使用清华 PyPI 源，失败时尝试官方源。可以在启动脚本前设置 `PIP_INDEX_URL`。基础模型默认先试 Hugging Face 镜像，再试官方站点，也可设置 `HF_ENDPOINT`。手动下载时请保留下面的文件名和路径。导出脚本会验证快照的 SHA-256。

| 源码目录下的位置 | 固定快照 |
| --- | --- |
| `assets/sources/contentvec/config.json` | [ContentVec 配置](https://huggingface.co/lengyue233/content-vec-best/resolve/ab04aa7067b99ee05cc82499bc64916b980a1967/config.json) |
| `assets/sources/contentvec/pytorch_model.bin` | [ContentVec 权重](https://huggingface.co/lengyue233/content-vec-best/resolve/ab04aa7067b99ee05cc82499bc64916b980a1967/pytorch_model.bin) |
| `assets/sources/rmvpe.pt` | [RMVPE 权重](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/e6d0c1a17da07c33557852f9dfa2bd44cc75737d/rmvpe.pt) |

## 调节与排查

- **没声音**　确认输入电平会随讲话变化，当前音色已选中，主输出已选对。聊天软件里的输入应为 `CABLE Output`。如果设备被独占或拔掉，关闭占用它的应用，停止变声后刷新设备，再重新开始。
- **有回声**　关闭「自己监听」，或佩戴耳机并降低监听音量。聊天软件的播放设备保持为耳机或音箱。
- **延迟偏大**　优先尝试「低延迟」，在设置里确认当前显卡是 RTX 3070 或另一张硬件显卡。状态栏显示估算链路延迟，聊天软件和网络的额外延迟由它们各自决定。
- **声音断续或游戏掉帧**　查看状态栏的推理耗时。出现性能提示时，可试用「均衡」或「高质量」，并为显卡留出余量。CPU 模式适合排查问题及离线转换。
- **没有索引或检索不可用**　音色仍可变声。需要音色贴合时，导入与该音色配套的索引，优先选 `added_` 开头的文件。
- **窗口无法打开**　确认系统可使用 Microsoft Edge WebView2 Runtime，查看程序目录下的 `logs/maskvox.log`。如果日志显示 ONNX Runtime 无法加载 DLL，可检查 x64 Visual C++ 运行时。 [WebView2 运行时](https://developer.microsoft.com/microsoft-edge/webview2/) · [Visual C++ 运行时](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

音色保存在 `voices/`，设置写入 `settings.json`。下次启动会恢复选择和参数，需手动开启变声。

MaskVox 1.0.0 便携 ZIP 约 416 MB，解压后运行文件约 598 MB，使用通过校验的 FP16 基础模型。

## 离线转换与打包

离线 WAV 转换使用与实时变声相同的分块、音高和 SOLA 拼接。调用时用音色库中的八位文件夹 ID。添加 `--gpu -1` 可选择 CPU。

```powershell
.venv\Scripts\python.exe -m scripts.offline_convert --voice abcd1234 --in a.wav --out b.wav --pitch 0
```

以下命令生成用于复现便携版的 FP16 基础模型和发行包。初次转换生成的 FP32 文件仍会保存在 `assets/`，便于对照。精简发行包支持导入 `.maskvox`，适合分发给无需转换社区检查点的用户。

```powershell
converter\.venv\Scripts\python.exe -m converter.export_base_models --fp16-out assets/fp16-candidate
.venv\Scripts\python.exe scripts\build_release.py --assets assets/fp16-candidate --out dist/MaskVox-local --zip
```

## 验证记录

本机使用 Windows 和 NVIDIA RTX 3070。使用 30 秒合成语音、真实 v2 音色及便携版的 FP16 基础模型，测得以下 DirectML 分块推理时间。FP16 基础模型另外通过了 CPU 数值校验和实际 DirectML 推理校验。

| 延迟档位 | 音频块 | 推理中位数 | 推理 P95 |
| --- | ---: | ---: | ---: |
| 低延迟 | 120 ms | 32.3 ms | 33.6 ms |
| 均衡 | 200 ms | 33.1 ms | 36.5 ms |
| 高质量 | 350 ms | 35.3 ms | 37.6 ms |

使用最终 FP16 管线和 HyperX 耳机路径，低延迟档的估算值中位数为 278 ms，P95 为 285 ms。设备缓冲与负载会影响估算值。测试有数次欠载回调，输入无溢出。HyperX 麦克风至耳机的独立直通试验连续 601 秒，欠载与溢出均为 0。

在 30 分钟实时压力测试中，以合成语音模拟设备时钟，真实 v2 音色通过 RTX 3070 的 DirectML 和 FP32 基础模型持续运行。均衡档推理中位数为 63.9 ms，P95 为 72.7 ms。进程工作集约 1.28 GiB，输入无溢出，输出缓冲记录 7 次欠载。

已用真实 v1 和 v2 音色完成转换与两种长度的验证，覆盖 `.maskvox` 导入、重新导出及重导。基础模型与真实音色的 ONNX 输出均对照 PyTorch 验证。另有自动化测试覆盖三种采样率的转换、损坏或不适用的检查点、恶意压缩包路径、ONNX 外部数据引用、合成模型分块、响度门限、音高、检索、缓冲及拼接。

在 30 秒真实 v2 音色转换对照中，MaskVox FP32 与同噪声的原生 PyTorch 管线相比，输出信噪比为 80.17 dB。与上游默认噪声增益的结果相比，共同判定为有声的帧音高误差中位数为 0 音分，P95 为 20 音分。测试使用合成语音，原始音色及输出样本未随便携版分发。

桌面界面测试覆盖本地文件启动、音色导入、实时启停和电平。用真实 v1/v2 模型及模拟音频设备测试，运行中切换音色用时约 2.5 秒。热调变调和关窗后恢复选择均通过验证。

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
converter\.venv\Scripts\python.exe -m unittest discover -s converter/tests -v
```

### 待真机验证

- 当前机器没有 VB-Cable。通过 CABLE Input 和 CABLE Output、聊天软件与远端听者测量端到端物理延迟和听感。请在该链路上复测 30 分钟连续运行，并观察设备欠载。
- 在另一台未安装 Python 的 Windows 电脑上复验便携版。当前机器已从独立的内嵌 Python 运行时测试。

第三方组件及权重来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
