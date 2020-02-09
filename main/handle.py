import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import torch
import numpy as np
import uuid


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'N/A','N/A','handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball','kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket','bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza','donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table','N/A', 'N/A', 'toilet', 'N/A',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone','microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def plot_preds(numpy_img, preds):
  boxes = preds['boxes'].detach().numpy()
  labels = preds['labels'].detach().numpy()
  for i in range(len(boxes)):
    box = boxes[i]
    label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    numpy_img = cv2.rectangle(
      numpy_img,
      (x1, y1),
      (x2, y2),
      255,
      3
    )
    cv2.putText(numpy_img, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
    # numpy_img = cv2.rectangle(
    #     numpy_img,
    #     (100,100),
    #     (200,200),
    #     255,
    #     3
    # )
  return numpy_img

def handle_uploaded_file(file):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
  uploaded_filename = file.name
  with open('main/static/main/images/1.jpeg', 'wb+') as destination:
    for chunk in file.chunks():
      destination.write(chunk)
  img = cv2.imread('main/static/main/images/1.jpeg')[:,:,::-1]
  img_numpy = img.astype('float32')
  img = torch.from_numpy(img_numpy.astype('float32')).permute(2,0,1)
  img = img/255.
  predictions = model(img[None,...])
  CONF_THRESH = 0.5
  boxes = predictions[0]['boxes'][predictions[0]['scores'] > CONF_THRESH]
  labels = predictions[0]['labels'][predictions[0]['scores'] > CONF_THRESH]
  boxes_dict = {}
  boxes_dict['boxes'] = boxes
  boxes_dict['labels'] = labels
  img_with_boxes = plot_preds(img_numpy, boxes_dict)
  unique_filename = str(uuid.uuid4())
  cv2.imwrite(f"main/static/main/images/{unique_filename}.jpeg", img_with_boxes[:,:,::-1])
  return unique_filename

