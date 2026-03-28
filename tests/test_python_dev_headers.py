from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from active_perception_r1.utils.preflight import require_python_dev_headers
from active_perception_r1.utils.python_dev_headers import (
    PythonDevHeaderStatus,
    ensure_python_dev_headers,
    format_exports,
    inspect_python_dev_headers,
)


class PythonDevHeadersTests(unittest.TestCase):
    def test_inspect_python_dev_headers_uses_env_include_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            include_dir = Path(temp_dir)
            (include_dir / "Python.h").write_text("/* test */", encoding="utf-8")

            status = inspect_python_dev_headers(
                system_include_dir=include_dir / "missing",
                env={"CPATH": str(include_dir)},
            )

            self.assertIsNotNone(status)
            assert status is not None
            self.assertEqual(status.source, "environment")
            self.assertEqual(status.include_dir, include_dir)
            self.assertEqual(status.compiler_include_dirs, (include_dir,))

    def test_ensure_python_dev_headers_extracts_vendored_deb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            vendor_root = repo_root / ".vendor_runtime" / "python312-dev"
            vendor_root.mkdir(parents=True)
            deb_path = vendor_root / "libpython3.12-dev_3.12.3-test_amd64.deb"
            deb_path.write_bytes(b"fake")

            extracted_paths: list[tuple[Path, Path]] = []

            def fake_extract(source: Path, extract_root: Path) -> None:
                extracted_paths.append((source, extract_root))
                include_dir = extract_root / "usr" / "include" / "python3.12"
                include_dir.mkdir(parents=True, exist_ok=True)
                (include_dir / "Python.h").write_text("/* vendored */", encoding="utf-8")
                multiarch_dir = extract_root / "usr" / "include" / "x86_64-linux-gnu" / "python3.12"
                multiarch_dir.mkdir(parents=True, exist_ok=True)
                (multiarch_dir / "pyconfig.h").write_text("/* pyconfig */", encoding="utf-8")

            status = ensure_python_dev_headers(
                repo_root=repo_root,
                system_include_dir=repo_root / "missing",
                env={},
                version_info=(3, 12),
                extract_fn=fake_extract,
            )

            self.assertEqual(status.source, "vendored")
            self.assertTrue((status.include_dir / "Python.h").is_file())
            self.assertEqual(extracted_paths[0][0], deb_path)
            self.assertEqual(
                status.compiler_include_dirs,
                (status.include_dir, status.include_dir.parent),
            )

    def test_require_python_dev_headers_raises_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_include = Path(temp_dir) / "missing"

            with self.assertRaises(RuntimeError):
                require_python_dev_headers(
                    purpose="smoke_train",
                    env={},
                    system_include_dir=str(missing_include),
                )

    def test_format_exports_is_shell_safe(self) -> None:
        status = PythonDevHeaderStatus(
            include_dir=Path("/tmp/fake include"),
            source="vendored",
            compiler_include_dirs=(Path("/tmp/fake include"), Path("/tmp/root include")),
        )
        exports = format_exports(status)
        self.assertIn("ACTIVE_PERCEPTION_PYTHON_INCLUDE_DIR", exports)
        self.assertIn("ACTIVE_PERCEPTION_PYTHON_INCLUDE_SOURCE", exports)
        self.assertIn("ACTIVE_PERCEPTION_PYTHON_COMPILER_INCLUDE_DIRS", exports)


if __name__ == "__main__":
    unittest.main()
