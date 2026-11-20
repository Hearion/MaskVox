from __future__ import annotations

import ctypes
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import onnxruntime as ort

logger = logging.getLogger(__name__)
DML = "DmlExecutionProvider"
CPU = "CPUExecutionProvider"


def available_providers() -> list[str]:
    return ort.get_available_providers()


def create_session(
    path: Path, gpu_index: int, fixed_dimensions: dict[str, int] | None = None, *,
    from_bytes: bool = False,
) -> ort.InferenceSession:
    if not path.is_file():
        raise RuntimeError(f"找不到模型文件 {path.name}，请检查基础模型或重新导入音色")
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if os.environ.get("MASKVOX_ORT_VERBOSE") == "1":
        options.log_severity_level = 0
    for dimension, size in (fixed_dimensions or {}).items():
        options.add_free_dimension_override_by_name(dimension, size)

    providers: list = [CPU]
    if gpu_index >= 0 and DML in available_providers():
        providers.insert(0, (DML, {"device_id": str(gpu_index)}))
    # Voice models are loaded from bytes to deny ONNX access to adjacent files.
    model = path.read_bytes() if from_bytes or not str(path).isascii() else str(path)
    session = ort.InferenceSession(model, sess_options=options, providers=providers)
    if DML in session.get_providers():
        # A failing provider is handled once for the pipeline during warm-up.
        session.disable_fallback()
    logger.info("Model %s providers %s", path.name, session.get_providers())
    return session


class _AdapterDesc(ctypes.Structure):
    _fields_ = [
        ("description", ctypes.c_wchar * 128),
        ("vendor_id", ctypes.c_uint),
        ("device_id", ctypes.c_uint),
        ("subsystem_id", ctypes.c_uint),
        ("revision", ctypes.c_uint),
        ("dedicated_video_memory", ctypes.c_size_t),
        ("dedicated_system_memory", ctypes.c_size_t),
        ("shared_system_memory", ctypes.c_size_t),
        ("adapter_luid", ctypes.c_uint32 * 2),
    ]


def _com_method(instance, index: int, result, *arguments):
    vtable = ctypes.cast(instance, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    method = ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(vtable[index])
    return lambda *values: method(instance, *values)


@contextmanager
def _owned_com(instance):
    try:
        yield instance
    finally:
        _com_method(instance, 2, ctypes.c_ulong)()  # IUnknown::Release


def list_adapters() -> list[dict]:
    """List adapters by the DXGI index that DirectML uses as device_id."""
    if os.name != "nt" or DML not in available_providers():
        return []

    adapters = []
    try:
        create_factory = ctypes.WinDLL("dxgi").CreateDXGIFactory
        create_factory.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
        create_factory.restype = ctypes.c_long
        iid = (ctypes.c_ubyte * 16).from_buffer_copy(
            UUID("7b7166ec-21c7-44ae-b21a-c9ae321ae369").bytes_le,
        )
        factory = ctypes.c_void_p()
        if create_factory(ctypes.byref(iid), ctypes.byref(factory)) < 0:
            raise OSError("CreateDXGIFactory failed")
        with _owned_com(factory):
            enumerate_adapter = _com_method(
                factory, 7, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
            )
            for index in range(32):
                adapter = ctypes.c_void_p()
                if enumerate_adapter(index, ctypes.byref(adapter)) < 0:
                    break
                with _owned_com(adapter):
                    description = _AdapterDesc()
                    get_desc = _com_method(adapter, 8, ctypes.c_long, ctypes.POINTER(_AdapterDesc))
                    if get_desc(ctypes.byref(description)) == 0:
                        adapters.append({"id": index, "name": description.description})
    except (OSError, AttributeError):
        logger.exception("Could not enumerate DirectML adapters")
    return adapters
