import os
import urllib.request
import shutil
from mrcnn.tools.config import Config


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

ROOT_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "images")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
# Local path to trained weights file
MODEL_DIR = os.path.join(ROOT_DIR, "weights")
if not os.path.exists(COCO_MODEL_DIR):
    os.makedirs(COCO_MODEL_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco.h5")

def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    MODEL_URL = "https://pjreddie.com/media/files/yolov3.weights"
    if verbose > 0:
        print("Downloading pretrained model to " + MODEL_PATH + " ...")
    with urllib.request.urlopen(MODEL_URL) as resp, open(MODEL_PATH, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")

# Download COCO trained weights from Releases if needed
if not os.path.exists(MODEL_PATH):
    download_trained_weights(MODEL_PATH, verbose=VERBOSE)

import GPUtil

LEAST_GMEM = 2250  # MB
MAX_THREADS = 1
MIN_FRAC = 0.3
MAX_FRAC = 0.3
GPU_LAOD = 0.5
GMEM_LAOD_LIMIT = 1.0
AVAIL_DEVICE_LIST = []
AVAIL_DEVICE_MAT = []
AVAIL_DEVICE_MEMFRAC = []
AVAIL_DEVICE_MAXTHREAD = []
try:
    GPUs = GPUtil.getGPUs()
    Gall = ''
    Gfree = ''
    for GPU in GPUs:
        Gall = GPU.memoryTotal
        Gfree = GPU.memoryFree
        GMEM_LAOD_LIMIT = float(format(float(LEAST_GMEM / Gall), '.2f'))
        if int(GPUtil.getAvailability([GPU], maxLoad=GPU_LAOD, maxMemory=GMEM_LAOD_LIMIT)) == 1:
            AVAIL_DEVICE_LIST.append(GPU)
            if GMEM_LAOD_LIMIT < MIN_FRAC:
                GMEM_LAOD_LIMIT = MIN_FRAC
            if GMEM_LAOD_LIMIT > MAX_FRAC:
                GMEM_LAOD_LIMIT = MAX_FRAC
            AVAIL_DEVICE_MEMFRAC.append(GMEM_LAOD_LIMIT)
            AVAIL_DEVICE_MAXTHREAD.append(int(1.0/GMEM_LAOD_LIMIT))
except Exception as e:
    print(e)

# initialize Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

BATCH_SIZE = 1
# initialize constants used for server queuing
IMAGE_QUEUE = "yolo3_queue"

SERVER_SLEEP = 0.1
CLIENT_SLEEP = 0.1

# Output Throttle
THROTTLE = 0.9
