import cv2
import torch
import torchvision

print("Creating model")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

src_img = cv2.imread('./lab9data/face.jpg')
img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
inputs = [img_tensor]

out = model(inputs)
boxes = out[0]['boxes']
scores = out[0]['scores']

threshold = 0.5
for idx in range(boxes.shape[0]):
    if scores[idx] >= threshold:
        x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
        cv2.rectangle(src_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

cv2.imshow('Detection Result', src_img)
cv2.waitKey()
cv2.destroyAllWindows()
