#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils import preprocess_input, decode_netout, correct_yolo_boxes, do_nms
from bbox import draw_boxes
from keras.models import load_model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Predict with a trained yolo model')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Set some parameter
    ###############################        
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.5, 0.45


    ###############################
    #   Load the model
    ###############################
    infer_model = load_model(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]
        """video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            
            # run the prediction
            yolos = infer_model.predict(new_image)
            boxes = []

            for i in range(len(yolos)):
                # decode the output of the network
                boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

            # correct the sizes of the bounding boxes
            correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

            # suppress non-maximal boxes
            do_nms(boxes, nms_thresh)     

            # draw bounding boxes on the image using labels
            draw_boxes(image, boxes, labels, obj_thresh) 

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()  """
    else:
        image = cv2.imread(image_path)

        # preprocess the input
        image_h, image_w, _ = image.shape
        new_image = preprocess_input(image, net_h, net_w)        

        # run the prediction
        yolos = infer_model.predict(new_image)
        boxes = []

        for i in range(len(yolos)):
            # decode the output of the network
            anchors = config['model']['anchors'][(2-i)*6:(3-i)*6]
            boxes += decode_netout(yolos[i][0], anchors, obj_thresh, nms_thresh, net_h, net_w)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)     

        # draw bounding boxes on the image using labels
        draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 
     
        # write the image with bounding boxes to file
        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], (image).astype('uint8')) 

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
