from __future__ import annotations

import argparse
import dataclasses
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


Runner = Callable[..., subprocess.CompletedProcess[str]]
Extractor = Callable[[Path, Path], None]


@dataclass(frozen=True)
class PythonDevHeaderStatus:
    include_dir: Path
    source: str
    compiler_include_dirs: tuple[Path, ...] = dataclasses.field(default_factory=tuple)


def python_version_tag(version_info: Sequence[int] | None = None) -> str:
    version_info = version_info or sys.version_info
    major, minor = int(version_info[0]), int(version_info[1])
    return f"python{major}.{minor}"


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def header_exists(include_dir: str | Path) -> bool:
    return Path(include_dir, "Python.h").is_file()


def derive_compiler_include_dirs(include_dir: str | Path) -> tuple[Path, ...]:
    include_path = Path(include_dir)
    candidates = [include_path]
    parent = include_path.parent
    if any(parent.glob(f"*/{include_path.name}/pyconfig.h")):
        candidates.append(parent)
    return tuple(_dedupe_paths(candidates))


def get_system_python_include_dir() -> Path:
    return Path(sysconfig.get_paths()["include"])


def iter_env_include_dirs(env: Mapping[str, str] | None = None) -> list[Path]:
    env = env or os.environ
    include_dirs: list[Path] = []
    for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        raw_value = env.get(key, "")
        for item in raw_value.split(os.pathsep):
            if item.strip():
                include_dirs.append(Path(item.strip()))
    return _dedupe_paths(include_dirs)


def inspect_python_dev_headers(
    *,
    system_include_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    extra_include_dirs: Sequence[str | Path] | None = None,
) -> PythonDevHeaderStatus | None:
    system_path = Path(system_include_dir) if system_include_dir is not None else get_system_python_include_dir()
    if header_exists(system_path):
        return PythonDevHeaderStatus(
            include_dir=system_path,
            source="system",
            compiler_include_dirs=derive_compiler_include_dirs(system_path),
        )

    for include_dir in iter_env_include_dirs(env):
        if header_exists(include_dir):
            return PythonDevHeaderStatus(
                include_dir=include_dir,
                source="environment",
                compiler_include_dirs=derive_compiler_include_dirs(include_dir),
            )

    for include_dir in extra_include_dirs or ():
        include_path = Path(include_dir)
        if header_exists(include_path):
            return PythonDevHeaderStatus(
                include_dir=include_path,
                source="extra",
                compiler_include_dirs=derive_compiler_include_dirs(include_path),
            )

    return None


def default_vendor_root(repo_root: str | Path, version_info: Sequence[int] | None = None) -> Path:
    version_info = version_info or sys.version_info
    major, minor = int(version_info[0]), int(version_info[1])
    return Path(repo_root) / ".vendor_runtime" / f"python{major}{minor}-dev"


def find_vendored_python_dev_deb(
    repo_root: str | Path,
    *,
    version_info: Sequence[int] | None = None,
) -> Path | None:
    version_tag = python_version_tag(version_info)
    vendor_root = Path(repo_root) / ".vendor_runtime"
    pattern = f"lib{version_tag}-dev_*.deb"
    matches = sorted(vendor_root.rglob(pattern))
    return matches[-1] if matches else None


def find_extracted_python_include_dir(
    extract_root: str | Path,
    *,
    version_info: Sequence[int] | None = None,
) -> Path | None:
    version_tag = python_version_tag(version_info)
    direct_candidate = Path(extract_root) / "usr" / "include" / version_tag
    if header_exists(direct_candidate):
        return direct_candidate

    for header_path in sorted(Path(extract_root).rglob("Python.h")):
        parent = header_path.parent
        if parent.name == version_tag:
            return parent
    return None


def extract_python_dev_headers_from_deb(
    deb_path: str | Path,
    extract_root: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> None:
    if shutil.which("dpkg-deb") is None:
        raise RuntimeError(
            "dpkg-deb is not available, so the vendored Python dev package cannot be extracted locally."
        )

    extract_root = Path(extract_root)
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    runner(
        ["dpkg-deb", "-x", str(deb_path), str(extract_root)],
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_python_dev_headers(
    *,
    repo_root: str | Path,
    system_include_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    version_info: Sequence[int] | None = None,
    extract_fn: Extractor | None = None,
) -> PythonDevHeaderStatus:
    resolution = inspect_python_dev_headers(
        system_include_dir=system_include_dir,
        env=env,
    )
    if resolution is not None:
        return resolution

    version_info = version_info or sys.version_info
    vendor_root = default_vendor_root(repo_root, version_info=version_info)
    extract_root = vendor_root / "extracted"

    extracted_include_dir = find_extracted_python_include_dir(
        extract_root,
        version_info=version_info,
    )
    if extracted_include_dir is not None:
        return PythonDevHeaderStatus(
            include_dir=extracted_include_dir,
            source="vendored",
            compiler_include_dirs=derive_compiler_include_dirs(extracted_include_dir),
        )

    deb_path = find_vendored_python_dev_deb(repo_root, version_info=version_info)
    if deb_path is None:
        version_tag = python_version_tag(version_info)
        raise RuntimeError(
            "Python development headers are unavailable for Triton/vLLM helper compilation. "
            f"Missing `Python.h` under `{get_system_python_include_dir()}` and no vendored "
            f"`lib{version_tag}-dev_*.deb` was found under `{Path(repo_root) / '.vendor_runtime'}`."
        )

    extract = extract_fn or (lambda src, dst: extract_python_dev_headers_from_deb(src, dst))
    extract(deb_path, extract_root)
    extracted_include_dir = find_extracted_python_include_dir(
        extract_root,
        version_info=version_info,
    )
    if extracted_include_dir is None:
        raise RuntimeError(
            f"Extracted `{deb_path}` but still could not locate `Python.h` under `{extract_root}`."
        )
    return PythonDevHeaderStatus(
        include_dir=extracted_include_dir,
        source="vendored",
        compiler_include_dirs=derive_compiler_include_dirs(extracted_include_dir),
    )


def format_exports(status: PythonDevHeaderStatus) -> str:
    return "\n".join(
        [
            f"export ACTIVE_PERCEPTION_PYTHON_INCLUDE_DIR={shlex.quote(str(status.include_dir))}",
            f"export ACTIVE_PERCEPTION_PYTHON_INCLUDE_SOURCE={shlex.quote(status.source)}",
            "export ACTIVE_PERCEPTION_PYTHON_COMPILER_INCLUDE_DIRS="
            + shlex.quote(os.pathsep.join(str(path) for path in status.compiler_include_dirs)),
        ]
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Resolve usable Python development headers for Triton builds.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--format",
        choices=("text", "exports"),
        default="text",
        help="Output format for resolved header information.",
    )
    args = parser.parse_args()

    status = ensure_python_dev_headers(repo_root=Path(args.repo_root).resolve())
    if args.format == "exports":
        print(format_exports(status))
    else:
        print(status.include_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
