
#Cloneing yolov5
!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
%pip install -qr requirements.txt  # install

#importing pytorch
import torch
from yolov5 import utils
display = utils.notebook_init()  # checks

!unzip -q ../train_data.zip -d../ #our file was a zip file

#Train YOLOv5s on COCO128 for 15 epochs
!python train.py --img 640 --batch 30 --epochs 20 --data coco128.yaml --weights yolov5s.pt --cache

#detecting
!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source /content/city.jpeg --save-crop

#installing pytesseract
!pip install pytesseract
!sudo apt install tesseract-ocr

#importing pytesseract and cv2
import pytesseract
import cv2

#extracting the text from the image
text = pytesseract.image_to_string("/content/yolov5/runs/detect/exp/crops/numberplate/city.jpg")
print(text)


