"""Read-only reuse of sibling projects without importing their CLI trainers.

Each training process activates one dependency family. This avoids the shared
`scripts` package name colliding across imu_semg and rgb_d. IMU and sEMG may
share a process; video uses a separate process. Python bytecode writes are
explicitly disabled for this process and spawned data-loader workers.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from macro_tools.paths import PROJECT_ROOT

_ACTIVE = None


def activate_source(modality):
    global _ACTIVE
    if modality not in ("imu", "semg", "rgb", "depth"):
        raise ValueError("Unknown modality: " + str(modality))
    family = "imu_semg" if modality in ("imu", "semg") else "rgb_d"
    root = (PROJECT_ROOT.parent / family).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Required sibling project is missing: " + str(root))
    if _ACTIVE is not None and _ACTIVE != family:
        raise RuntimeError("Use a separate Python process for each source-project family")
    for package in ("scripts", "imu", "semg") if family == "imu_semg" else ("scripts", "vision"):
        module = sys.modules.get(package)
        location = getattr(module, "__file__", None)
        if location and not Path(location).resolve().is_relative_to(root):
            raise RuntimeError("Conflicting package " + package + "; start a fresh training process")
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _ACTIVE = family
    return root


def source_versions():
    """Hash actual loaded source files, including uncommitted model changes."""
    versions = {}
    for family in ("imu_semg", "rgb_d"):
        root = (PROJECT_ROOT.parent / family).resolve()
        files = {}
        for module in list(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if not filename:
                continue
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(root) and path.is_file():
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        if files:
            try:
                revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                revision = None
            versions[family] = {"relative_path": "../" + family, "git_revision": revision, "loaded_file_sha256": files}
    return versions
