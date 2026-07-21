import sys, os, pathlib
import onnxruntime as ort
print("python:", sys.executable)
print("onnxruntime:", ort.__version__)
print("onnxruntime __file__:", ort.__file__)
print("available providers:", ort.get_available_providers())
capi_dir = pathlib.Path(ort.__file__).parent / "capi"
print("onnxruntime capi dir exists:", capi_dir.exists())
if capi_dir.exists():
    print([p.name for p in capi_dir.iterdir() if p.suffix.lower() in ('.dll','.so')][:20])
PATH = os.environ.get("PATH","").split(os.pathsep)
for i,p in enumerate(PATH[:20]):
    print(f"PATH[{i}]:", p)
for dll in ("cudnn64_9.dll","cudnn64_8.dll","cudnn64_7.dll"):
    found = [str(pathlib.Path(p)/dll) for p in PATH if (pathlib.Path(p)/dll).exists()]
    print(dll, "found at:", found)
