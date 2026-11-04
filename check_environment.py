"""Check the one-environment requirements for all three sibling projects.

Read-only checks only. Exits nonzero when a required import is missing. CUDA
availability is reported; CPU use remains valid. No packages are installed.
"""
import importlib
import os
import sys
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

def main():
    print("Python:", sys.executable, sys.version.split()[0])
    missing = []
    for name in ("numpy", "scipy", "torch", "h5py", "cv2", "matplotlib"):
        try:
            module = importlib.import_module(name)
            print(name + ":", getattr(module, "__version__", "import OK"))
        except ImportError as error:
            missing.append(name)
            print(name + ": MISSING -", error)
    if "torch" not in missing:
        import torch
        print("CUDA build:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    from macro_tools.paths import PROJECT_ROOT
    for name in ("dataset", "imu_semg", "rgb_d"):
        path = PROJECT_ROOT.parent / name
        print("Sibling", name + ":", "found" if path.is_dir() else "MISSING")
        if not path.is_dir():
            missing.append(name)
    return int(bool(missing))

if __name__ == "__main__":
    raise SystemExit(main())
