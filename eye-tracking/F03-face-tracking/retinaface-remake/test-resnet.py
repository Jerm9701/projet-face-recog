import torchvision.models as models
import torch
from torchvision.models import ResNet50_Weights
import torchvision.transforms as T
import cv2

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

img_path = "captures/2.png"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")

# Convert BGR -> RGB and build the tensor with pretrained normalization.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32) / 255.0
weights = ResNet50_Weights.DEFAULT
transform = T.Compose([
    T.Resize(weights.transforms().resize_size),
    T.CenterCrop(weights.transforms().crop_size),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])

# Apply the torchvision transforms to the tensor and add batch dimension
img_tensor = transform(img_tensor).unsqueeze(0)

with torch.no_grad():
    logits = model(img_tensor)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    top5_probs, top5_idxs = probabilities.topk(5, dim=1)

print("Top-5 predictions:")
for prob, idx in zip(top5_probs[0], top5_idxs[0]):
    class_name = weights.meta["categories"][idx.item()]
    print(f"Class {idx.item():3d} | {class_name:20s} | {prob.item():.4f}")