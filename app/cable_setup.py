"""Verify and open the bundled VB-Audio VB-CABLE installer."""

from __future__ import annotations

import ctypes
import hashlib
import io
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import zipfile
from ctypes import wintypes
from pathlib import Path

from .config import ASSETS_DIR

CABLE_ARCHIVE = ASSETS_DIR / "drivers" / "VBCABLE_Driver_Pack46.zip"
CABLE_ARCHIVE_URL = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack46.zip"
CABLE_ARCHIVE_SHA256 = "bc44e8974191a5a2427d6c979e2dab08dbb09925f3f435dfa2f0ba3e45b6eb90"
CABLE_INSTALLER = "VBCABLE_Setup_x64.exe"
CABLE_INSTALLER_SHA256 = "734c35dfa6d98f48782a451633ceb471166ec70d60482fd89a1123d0ee3c4f41"
CABLE_CATALOG = "vbaudio_cable64_win10.cat"
CABLE_CATALOG_SHA256 = "5d24c17858b49ad83bb856c66c24c5c2aecdb89cbaae2f8179c15d351124fa9f"

_REQUIRED = {CABLE_INSTALLER, CABLE_CATALOG, "vbMmeCable64_win10.inf", "vbaudio_cable64_win10.sys"}
_ARCHIVE_LIMIT = 4 * 1024**2
_EXTRACTED_LIMIT = 8 * 1024**2
_SAFE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,94}\.(?:cat|exe|ico|inf|sys|txt)", re.IGNORECASE)
_RESERVED = re.compile(r"(?:con|prn|aux|nul|com[0-9]|lpt[0-9])", re.IGNORECASE)
_LAUNCH_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)


def verify_installer_archive(path: Path | None = None) -> bytes:
    """Read a bounded, checksum-verified snapshot of the official ZIP."""
    archive = CABLE_ARCHIVE if path is None else path
    try:
        with archive.open("rb") as stream:
            data = stream.read(_ARCHIVE_LIMIT + 1)
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 VB-CABLE 离线安装包。请从官方页面获取驱动。") from exc
    except OSError as exc:
        raise RuntimeError("无法读取 VB-CABLE 离线安装包。") from exc
    if len(data) > _ARCHIVE_LIMIT or hashlib.sha256(data).hexdigest() != CABLE_ARCHIVE_SHA256:
        raise RuntimeError("VB-CABLE 离线安装包校验失败。请重新获取官方安装包。")
    return data


def is_installer_available() -> bool:
    try:
        verify_installer_archive()
    except RuntimeError:
        return False
    return True


def _extract_installer(data: bytes, directory: Path) -> Path:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 64 or sum(entry.file_size for entry in entries) > _EXTRACTED_LIMIT:
                raise RuntimeError("VB-CABLE 安装包的文件数量或大小不受支持。")
            names = set()
            for entry in entries:
                name = entry.orig_filename
                mode = stat.S_IFMT(entry.external_attr >> 16)
                if (
                    not _SAFE_NAME.fullmatch(name)
                    or _RESERVED.fullmatch(Path(name).stem)
                    or mode not in (0, stat.S_IFREG)
                    or entry.flag_bits & 1
                    or name.casefold() in names
                ):
                    raise RuntimeError("VB-CABLE 安装包包含不安全的文件条目。")
                names.add(name.casefold())
            if not _REQUIRED.issubset(archive.namelist()):
                raise RuntimeError("VB-CABLE 安装包缺少安装器或驱动文件。")
            for entry in entries:
                # All entries are small flat regular files, and the destination is new.
                with (directory / entry.filename).open("xb") as target:
                    target.write(archive.read(entry))
    except (OSError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise RuntimeError("无法解压 VB-CABLE 离线安装包。") from exc
    return directory / CABLE_INSTALLER


class _ShellExecuteInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD), ("fMask", wintypes.ULONG), ("hwnd", wintypes.HWND),
        ("lpVerb", wintypes.LPCWSTR), ("lpFile", wintypes.LPCWSTR), ("lpParameters", wintypes.LPCWSTR),
        ("lpDirectory", wintypes.LPCWSTR), ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE),
        ("lpIDList", ctypes.c_void_p), ("lpClass", wintypes.LPCWSTR), ("hkeyClass", wintypes.HKEY),
        ("dwHotKey", wintypes.DWORD), ("hIcon", wintypes.HANDLE), ("hProcess", wintypes.HANDLE),
    ]


def _run_elevated(executable: Path) -> int | None:
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    execute = shell.ShellExecuteExW
    execute.argtypes = [ctypes.POINTER(_ShellExecuteInfo)]
    execute.restype = wintypes.BOOL
    info = _ShellExecuteInfo(
        cbSize=ctypes.sizeof(_ShellExecuteInfo),
        fMask=0x00000440,  # NOCLOSEPROCESS | FLAG_NO_UI (UAC still appears)
        lpVerb="runas",
        lpFile=str(executable),
        lpDirectory=str(executable.parent),
        nShow=1,  # SW_SHOWNORMAL
    )
    ole32 = ctypes.WinDLL("ole32")
    result = ole32.CoInitializeEx(None, 6)  # APARTMENTTHREADED | DISABLE_OLE1DDE
    if result < 0 and result != ctypes.c_long(0x80010106).value:  # RPC_E_CHANGED_MODE
        raise RuntimeError("无法初始化 Windows 安装界面。")
    try:
        if not execute(ctypes.byref(info)):
            error = ctypes.get_last_error()
            if error == 1223:  # ERROR_CANCELLED
                raise RuntimeError("已取消 Windows 管理员授权。")
            raise RuntimeError(f"无法打开 VB-CABLE 安装器（Windows 错误 {error}）。")
        return info.hProcess
    finally:
        if result >= 0:
            ole32.CoUninitialize()


def _close_process(handle: int) -> None:
    close = ctypes.WinDLL("kernel32").CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    close(handle)


def _clean_after_exit(handle: int, directory: Path) -> None:
    try:
        wait = ctypes.WinDLL("kernel32").WaitForSingleObject
        wait.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        wait.restype = wintypes.DWORD
        if wait(handle, 0xFFFFFFFF) == 0:
            shutil.rmtree(directory, ignore_errors=True)
    finally:
        _close_process(handle)
        _LAUNCH_LOCK.release()


def launch_installer() -> None:
    """Open the stock setup with UAC. Only the cleanup thread waits for setup."""
    if os.name != "nt":
        raise RuntimeError("VB-CABLE 驱动安装器仅支持 Windows。")
    if not _LAUNCH_LOCK.acquire(blocking=False):
        raise RuntimeError("VB-CABLE 安装器已在打开，请检查 Windows 安装窗口。")

    directory = None
    launched = cleanup_started = False
    try:
        data = verify_installer_archive()
        directory = Path(tempfile.mkdtemp(prefix="MaskVox-VBCable-"))
        installer = _extract_installer(data, directory)
        if (
            hashlib.sha256(installer.read_bytes()).hexdigest() != CABLE_INSTALLER_SHA256
            or hashlib.sha256((directory / CABLE_CATALOG).read_bytes()).hexdigest() != CABLE_CATALOG_SHA256
        ):
            raise RuntimeError("VB-CABLE 安装器或驱动签名校验失败。")
        handle = _run_elevated(installer)
        launched = True
        if handle:
            worker = threading.Thread(
                target=_clean_after_exit, args=(handle, directory), name="vbcable-setup-cleanup", daemon=True,
            )
            try:
                worker.start()
                cleanup_started = True
            except RuntimeError:
                _close_process(handle)
                _LOGGER.warning("Could not schedule VB-CABLE setup cleanup", exc_info=True)
    except OSError as exc:
        raise RuntimeError("无法准备 VB-CABLE 安装器。") from exc
    finally:
        if not cleanup_started:
            if directory is not None and not launched:
                shutil.rmtree(directory, ignore_errors=True)
            _LAUNCH_LOCK.release()
