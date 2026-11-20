from __future__ import annotations

import hashlib
import io
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from app import cable_setup as setup


def _archive(*extra: tuple[str | zipfile.ZipInfo, bytes]) -> bytes:
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        for name in sorted(setup._REQUIRED):
            archive.writestr(name, b"fixture " + name.encode("ascii"))
        for name, data in extra:
            archive.writestr(name, data)
    return content.getvalue()


class DriverSetupTests(unittest.TestCase):
    def test_missing_or_tampered_zip_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "driver.zip"
            with patch.object(setup, "CABLE_ARCHIVE", path):
                self.assertFalse(setup.is_installer_available())
                with self.assertRaisesRegex(RuntimeError, "未找到"):
                    setup.verify_installer_archive()
                data = _archive()
                path.write_bytes(data)
                with patch.object(setup, "CABLE_ARCHIVE_SHA256", hashlib.sha256(data).hexdigest()):
                    self.assertTrue(setup.is_installer_available())
                    self.assertEqual(setup.verify_installer_archive(), data)
                    path.write_bytes(data + b"tampered")
                    self.assertFalse(setup.is_installer_available())
                    with self.assertRaisesRegex(RuntimeError, "校验失败"):
                        setup.verify_installer_archive()

    def test_extract_rejects_unsafe_entries_before_writing(self) -> None:
        symlink = zipfile.ZipInfo("shortcut.inf")
        symlink.create_system = 3
        symlink.external_attr = stat.S_IFLNK << 16
        for unsafe in ("../outside.exe", "C:/outside.exe", "..\\outside.exe", "NUL.txt", "vbcable_setup_x64.EXE", symlink):
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                with self.assertRaisesRegex(RuntimeError, "不安全"):
                    setup._extract_installer(_archive((unsafe, b"fixture")), directory)
                self.assertEqual(list(directory.iterdir()), [])

    def test_extract_requires_driver_files_and_bounds_total(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            content = io.BytesIO()
            with zipfile.ZipFile(content, "w") as archive:
                archive.writestr(setup.CABLE_INSTALLER, b"fixture")
            with self.assertRaisesRegex(RuntimeError, "缺少"):
                setup._extract_installer(content.getvalue(), directory)
            with patch.object(setup, "_EXTRACTED_LIMIT", 1):
                with self.assertRaisesRegex(RuntimeError, "大小"):
                    setup._extract_installer(_archive(), directory)
            self.assertEqual(list(directory.iterdir()), [])

    def test_launch_passes_verified_x64_exe_to_shell_and_does_not_wait(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "测试驱动"
            directory.mkdir()
            data = _archive()
            known_exe = b"fixture " + setup.CABLE_INSTALLER.encode("ascii")
            known_cat = b"fixture " + setup.CABLE_CATALOG.encode("ascii")
            with (
                patch.object(setup, "os") as os_proxy,
                patch.object(setup, "verify_installer_archive", return_value=data),
                patch.object(setup, "CABLE_INSTALLER_SHA256", hashlib.sha256(known_exe).hexdigest()),
                patch.object(setup, "CABLE_CATALOG_SHA256", hashlib.sha256(known_cat).hexdigest()),
                patch.object(setup.tempfile, "mkdtemp", return_value=str(directory)),
                patch.object(setup, "_run_elevated", return_value=123) as launch,
                patch.object(setup.threading, "Thread") as worker,
            ):
                os_proxy.name = "nt"
                setup.launch_installer()
                try:
                    executable = directory / setup.CABLE_INSTALLER
                    self.assertEqual(executable.read_bytes(), known_exe)
                    launch.assert_called_once_with(executable)
                    worker.assert_called_once_with(
                        target=setup._clean_after_exit, args=(123, directory), daemon=True,
                        name="vbcable-setup-cleanup",
                    )
                    worker.return_value.start.assert_called_once_with()
                    with self.assertRaisesRegex(RuntimeError, "已在打开"):
                        setup.launch_installer()
                finally:
                    # The mocked cleanup worker above never really starts.
                    setup._LAUNCH_LOCK.release()

    def test_cancelled_uac_releases_lock_and_removes_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "解压"
            directory.mkdir()
            data = _archive()
            with (
                patch.object(setup, "os") as os_proxy,
                patch.object(setup, "verify_installer_archive", return_value=data),
                patch.object(setup, "CABLE_INSTALLER_SHA256", hashlib.sha256(b"fixture VBCABLE_Setup_x64.exe").hexdigest()),
                patch.object(setup, "CABLE_CATALOG_SHA256", hashlib.sha256(b"fixture vbaudio_cable64_win10.cat").hexdigest()),
                patch.object(setup.tempfile, "mkdtemp", return_value=str(directory)),
                patch.object(setup, "_run_elevated", side_effect=RuntimeError("已取消 Windows 管理员授权。")),
            ):
                os_proxy.name = "nt"
                with self.assertRaisesRegex(RuntimeError, "已取消"):
                    setup.launch_installer()
                self.assertFalse(directory.exists())
                self.assertFalse(setup._LAUNCH_LOCK.locked())

    def test_windows_shell_uses_uac_and_no_install_arguments(self) -> None:
        shell = Mock()
        ole = Mock()
        ole.CoInitializeEx.return_value = 0
        executable = Path("drive setup") / setup.CABLE_INSTALLER

        def run(info_pointer) -> int:
            info = info_pointer._obj
            self.assertEqual(info.lpVerb, "runas")
            self.assertEqual(info.lpFile, str(executable))
            self.assertEqual(info.lpDirectory, str(executable.parent))
            self.assertIsNone(info.lpParameters)
            self.assertEqual(info.nShow, 1)
            return 0

        shell.ShellExecuteExW.side_effect = run
        with (
            patch.object(setup.ctypes, "WinDLL", side_effect=[shell, ole], create=True),
            patch.object(setup.ctypes, "get_last_error", return_value=1223, create=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "已取消 Windows 管理员授权"):
                setup._run_elevated(executable)
        ole.CoUninitialize.assert_called_once_with()
