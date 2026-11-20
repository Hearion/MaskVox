"""Built-in voice effects. Each one is a full set of chain parameters."""

from __future__ import annotations

from .vocoder import BANDS, MODES

# The four values the interface exposes as sliders.
ADJUSTABLE = {"pitch": (-12, 12, 0.5), "formant": (-12, 12, 0.5), "echo": (0, 1, 0.05), "reverb": (0, 1, 0.05)}
_BASE = {
    "pitch": 0, "formant": 0, "echo": 0, "reverb": 0,
    "mode": "normal", "band": "full", "tune": False, "ring_hz": 0, "tremolo": 0, "drive": 0,
}


def _preset(identifier: str, name: str, hint: str, icon: str, hue: int, **params) -> dict:
    unknown = params.keys() - _BASE.keys()
    if unknown or params.get("mode", "normal") not in MODES or params.get("band", "full") not in BANDS:
        raise ValueError(f"音效 {identifier} 的参数无效")
    return {"id": identifier, "name": name, "hint": hint, "icon": icon, "hue": hue, "params": {**_BASE, **params}}


PRESETS = (
    _preset("original", "原声", "不做处理", "🎙️", 210),
    _preset("woman", "女声", "男声变女声", "👩", 330, pitch=9, formant=3),
    _preset("loli", "萝莉", "又细又甜", "🎀", 345, pitch=12, formant=5),
    _preset("boy", "正太", "稚气少年", "🧒", 190, pitch=5, formant=2),
    _preset("man", "大叔", "女声变男声", "🧔", 28, pitch=-5, formant=-2.5),
    _preset("giant", "巨人", "低沉厚重", "🗿", 160, pitch=-9, formant=-5, reverb=0.15),
    _preset("demon", "恶魔", "嘶哑阴森", "😈", 0, pitch=-7, formant=-4, drive=0.35, reverb=0.25),
    _preset("chipmunk", "花栗鼠", "快放的磁带", "🐿️", 40, pitch=12, formant=9),
    _preset("robot", "机器人", "没有语调", "🤖", 200, mode="robot"),
    _preset("alien", "外星人", "金属颤音", "👽", 130, pitch=3, ring_hz=60),
    _preset("ghost", "幽灵", "气声加回响", "👻", 265, mode="whisper", formant=-1, reverb=0.45),
    _preset("tune", "电音", "音高吸附到音阶", "🎛️", 285, tune=True),
    _preset("phone", "电话", "老式听筒", "☎️", 50, band="phone", drive=0.15),
    _preset("radio", "对讲机", "失真加窄带", "📻", 15, band="radio", drive=0.5),
    _preset("cave", "山洞", "回声阵阵", "🏔️", 175, echo=0.4, reverb=0.5),
    _preset("underwater", "水下", "闷闷的咕噜声", "🌊", 220, band="muffled", tremolo=0.5, reverb=0.2),
)
BY_ID = {preset["id"]: preset for preset in PRESETS}


def chain_params(settings: dict) -> dict:
    """Parameters for the audio chain: the preset with the user's slider values."""
    params = dict(BY_ID.get(settings.get("preset"), PRESETS[0])["params"])
    params.update({key: settings[key] for key in ADJUSTABLE if key in settings})
    params["gate_db"] = settings.get("gate_db", -50)
    return params
