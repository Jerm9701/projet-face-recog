import onnxruntime as ort
import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device available")
print(ort.__version__)
print('providers', ort.get_available_providers())