import torchvision
import torch
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
import cv2
model =torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
model.eval()

img_path = "captures/rd_img1.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")

# Convert BGR -> RGB, then HWC -> CHW and normalize to float [0, 1]
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32) / 255.0

predictions = model([img_tensor])
boxes = predictions[0]["boxes"]
scores = predictions[0]["scores"]
keep = scores > 0.9


for box, score in zip(boxes[keep], scores[keep]):
    if score > 0.9:
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()