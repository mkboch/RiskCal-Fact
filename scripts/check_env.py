import os
import sys
import platform
import subprocess

print("==== Python ====")
print("Executable:", sys.executable)
print("Version:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())

print("\n==== Project ====")
print("CWD:", os.getcwd())
print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV", "NOT SET"))

print("\n==== NVIDIA-SMI ====")
try:
    out = subprocess.check_output(
        ["nvidia-smi"],
        stderr=subprocess.STDOUT,
        text=True
    )
    print(out)
except Exception as e:
    print("nvidia-smi failed:", repr(e))

print("\n==== Disk ====")
try:
    out = subprocess.check_output(
        ["bash", "-lc", "df -h $HOME . | tail -n +1"],
        stderr=subprocess.STDOUT,
        text=True
    )
    print(out)
except Exception as e:
    print("df failed:", repr(e))

print("\n==== Existing torch check ====")
try:
    import torch
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
except Exception as e:
    print("torch not installed or failed:", repr(e))
