import argparse
import os

import struct
import cv2
import numpy as np
from utils.weightreader import WeightReader
from utils.bbox import BoundBox
from utils.tools import preprocess_input, decode_netout
from utils.tools import correct_yolo_boxes, do_nms, draw_boxes
from model.yolo3 import make_yolov3_model

np.set_printoptions(threshold=np.nan)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# set some parameters
net_h, net_w = 416, 416
obj_thresh, nms_thresh = 0.5, 0.45
anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
            "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def post_stream(yolos, boxes, image):
    image_h, image_w, _ = image.shape
    for i in range(len(yolos)):
        # decode the output of the network
        boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, net_h, net_w)

    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)
    # draw bounding boxes on the image using labels
    i = draw_boxes(image, boxes, labels, obj_thresh)
    return i

def _main_(args):
    weights_path = args.weights

    # make the yolov3 model to predict 80 classes on COCO
    yolov3 = make_yolov3_model()

    # load the weights trained on COCO into the model
    weight_reader = WeightReader(weights_path)
    weight_reader.load_weights(yolov3)

    # While frame
    #   pre_stream()
    #   detect()
    #   post_stream()
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280) # set the Horizontal resolution
    cap.set(4, 720)
    while(True):
        # Capture frame-by-frame
        ret, image = cap.read()
        # preprocess the image
        new_image = preprocess_input(image, net_h, net_w)
        # run the prediction
        yolos = yolov3.predict(new_image)
        boxes = []
        frame = post_stream(yolos, boxes, image)
        cv2.imshow('Yolo3', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')
    argparser.add_argument(
        '-w',
        '--weights',
        help='path to weights file')

    args = argparser.parse_args()
    _main_(args)
