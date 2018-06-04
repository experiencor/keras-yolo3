import base64
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault('PATH', '')
import numpy as np
import redis
import time
import json
from io import BytesIO
from multiprocessing import Process, Pipe, current_process, Lock
import GPUtil
from skimage.measure import find_contours
import struct
import cv2
import numpy as np
import config

# connect to Redis server
redispool = redis.ConnectionPool(host=config.REDIS_HOST,
                          port=config.REDIS_PORT,
                          db=config.REDIS_DB,
                          socket_keepalive=True)

try:
    print('Testing Redis Connection')
    redisdbSession = redis.StrictRedis(connection_pool=redispool)
    response = redisdbSession.client_list()
    print('Redis Connection Established')
except redis.ConnectionError as e:
    print(e)
    sys.exit(1)

np.set_printoptions(threshold=np.nan)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# set some parameters
net_h, net_w = 416, 416
obj_thresh, nms_thresh = 0.7, 0.7
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

class mlWorker(Process):
    def __init__(self, LOCK, GPU="", FRAC=0):
        Process.__init__(self)
        self.lock = LOCK
        if GPU:
            print('{} using GPUid: {}, Name: {}'.format(self.name, str(GPU.id), str(GPU.name)))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU.id)
            self.device = '/device:GPU:0'
        else:
            self.device = ''
        self.GPU = GPU
        self.frac = FRAC
        self.counter = 0
        self.dt = 0.0

    def run(self):
        from utils.weightreader import WeightReader
        from utils.bbox import BoundBox
        from utils.tools import preprocess_input, decode_netout
        from utils.tools import correct_yolo_boxes, do_nms, draw_boxes
        from model.yolo3 import make_yolov3_model
        import tensorflow as tf
        from PIL import Image
        self.Image = Image
        self.preprocess_input = preprocess_input
        self.decode_netout = decode_netout
        self.correct_yolo_boxes = correct_yolo_boxes
        self.do_nms = do_nms
        self.draw_boxes = draw_boxes
        if self.GPU:
            print('ML Process: {} starting, using GPU: {}, frac: {}'.format(self.name,self.GPU.id,self.frac))
        keras.backend.clear_session()
        conf = tf.ConfigProto()
        conf.gpu_options.per_process_gpu_memory_fraction = self.frac
        set_session(tf.Session(config=conf))
        # make the yolov3 model to predict 80 classes on COCO
        _model = make_yolov3_model()

        # load the weights trained on COCO into the model
        weight_reader = WeightReader(config.MODEL_PATH)
        weight_reader.load_weights(_model)

        graph = tf.get_default_graph()
        print('ML Process: {} started'.format(self.name))
        self.mainloop(model=_model, graph=graph)

    def mainloop(self, model='', graph=''):
        while True:
            # attempt to grab a batch of images from the database, then
            # initialize the image IDs and batch of images themselves
            try:
                redisdbSession = redis.StrictRedis(connection_pool=redispool)
                self.lock.acquire()
                query = redisdbSession.lrange(config.IMAGE_QUEUE, 0, config.BATCH_SIZE - 1)
                redisdbSession.ltrim(config.IMAGE_QUEUE, len(query), -1)
                self.lock.release()
                imageIDs = []
                thresholds = {}
                batch = []
                # loop over the queue
                # deserialize the object and obtain the input image
                if query:
                    for item in query:
                        data = json.loads(item)
                        image = self.base64_decode_image(data["image"])
                        image = self.preprocess_input(image, net_h, net_w)
                        # check to see if the batch list is None
                        batch.append(image)
                        # update the list of image IDs
                        imageIDs.append(data["id"])
                        thresholds[data["id"]] = data["threshold"]

                # check to see if we need to process the batch
                if len(imageIDs) > 0:
                    #print('{}: Procesing {} images!'.format(self.name, len(imageIDs)))
                    start = time.time()
                    with graph.as_default():
                        results = model.predict(batch[0])
                    end = time.time()
                    et = end - start
                    self.dt += float(et)
                    self.counter += 1
                    adt = float(self.dt)/float(self.counter)
                    print('avg dt: %f' % adt) 
                    # loop over the image IDs and their corresponding set of
                    # results from our model
                    output = []
                    output = self.extract_result(results,
                        throttle=float(thresholds[imageID]))
                    redisdbSession.set(imageID, json.dumps(output))
                # sleep for a small amount
                time.sleep(config.SERVER_SLEEP*2)
            except Exception as e:
                print(e)
                time.sleep(config.SERVER_SLEEP)
                continue

    def extract_result(self, results, throttle='0.95'):
        boxes = []

        for i in range(len(yolos)):
            # decode the output of the network
            boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, net_h, net_w)
        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)     

        return output

    def base64_decode_image(self, a):
        """
        return: <ndarray>
        """
        img = self.Image.open(BytesIO(base64.b64decode(a)))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = np.array(img)
        return img



if __name__ == "__main__":
    LOCK = Lock()
    AVAIL_DEVICE_LIST = config.AVAIL_DEVICE_LIST
    AVAIL_DEVICE_MEMFRAC = config.AVAIL_DEVICE_MEMFRAC
    AVAIL_DEVICE_MAXTHREAD = config.AVAIL_DEVICE_MAXTHREAD

    proc_list = []
    print('{} GPUs Available'.format(len(AVAIL_DEVICE_LIST)))
    if AVAIL_DEVICE_LIST:
        for index, device in enumerate(AVAIL_DEVICE_LIST):
            thread_count = int(AVAIL_DEVICE_MAXTHREAD[index])
            mem_frac = float(AVAIL_DEVICE_MEMFRAC[index])
            if config.MAX_FRAC < mem_frac:
                mem_frac = config.MAX_FRAC
            print('Preparing {} process on GPU: {}, frac: {}'.format(thread_count, device.id, mem_frac))
            if config.MAX_THREADS < thread_count:
                thread_count = config.MAX_THREADS
            for thread in range(thread_count):
                p = mlWorker(LOCK, GPU=device, FRAC=mem_frac)
                p.daemon = True
                proc_list.append(p)
        print('Starting total: {} processes'.format(len(proc_list)))
        for proc in proc_list:
            proc.start()
        print('All processes started')
    else:
        p = mlWorker(LOCK)
        p.daemon = True
        p.start()
        p.join()

    if proc_list:
        for proc in proc_list:
            proc.join()
