from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence


_DEFAULT_BUILD_DIRS = ("build_cmake_sm120a", "build_cmake_sm100a", "build")
_ARCH_BUILD_DIRS = {
    10: "build_cmake_sm100a",
    12: "build_cmake_sm120a",
}


def _append_unique_path(paths: list[Path], seen: set[str], candidate: Path | str) -> None:
    path = Path(candidate).expanduser()
    path_str = os.fspath(path)
    if path_str in seen:
        return
    seen.add(path_str)
    paths.append(path)


def _preferred_build_dir_name() -> str | None:
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return None

    return _ARCH_BUILD_DIRS.get(major)


def get_preferred_build_dir_names() -> tuple[str, ...]:
    preferred_build_dir = _preferred_build_dir_name()
    if preferred_build_dir is None:
        return _DEFAULT_BUILD_DIRS

    remaining_build_dirs = tuple(
        build_dir_name for build_dir_name in _DEFAULT_BUILD_DIRS if build_dir_name != preferred_build_dir
    )
    return (preferred_build_dir, *remaining_build_dirs)


def get_sharq_build_candidates(
    repo_root: Path | None = None,
    extra_candidates: Iterable[Path | str] = (),
    env_vars: Sequence[str] = ("SHARQ_KERNEL_BUILD",),
) -> list[Path]:
    resolved_repo_root = repo_root if repo_root is not None else Path(__file__).resolve().parent
    candidates: list[Path] = []
    seen: set[str] = set()

    for env_var in env_vars:
        env_build = os.environ.get(env_var)
        if env_build:
            _append_unique_path(candidates, seen, env_build)

    for build_dir_name in get_preferred_build_dir_names():
        _append_unique_path(candidates, seen, resolved_repo_root / "kernels" / build_dir_name)

    for candidate in extra_candidates:
        _append_unique_path(candidates, seen, candidate)

    return candidates


def load_sharq_ops(
    repo_root: Path | None = None,
    module_names: Sequence[str] = ("sharq_ops",),
    extra_candidates: Iterable[Path | str] = (),
    env_vars: Sequence[str] = ("SHARQ_KERNEL_BUILD",),
):
    last_error: ImportError | None = None
    searched: list[str] = []

    for build_dir in get_sharq_build_candidates(
        repo_root=repo_root,
        extra_candidates=extra_candidates,
        env_vars=env_vars,
    ):
        build_dir_str = os.fspath(build_dir)
        searched.append(build_dir_str)

        if not build_dir.exists():
            continue

        if build_dir_str not in sys.path:
            sys.path.insert(0, build_dir_str)

        for module_name in module_names:
            try:
                return importlib.import_module(module_name)
            except ImportError as exc:
                last_error = exc

    searched_dirs = ", ".join(searched) if searched else "<none>"
    message = f"Failed to import any of {tuple(module_names)} from: {searched_dirs}"
    if last_error is not None:
        raise ImportError(message) from last_error
    raise ImportError(message)
