FROM python:3.7

WORKDIR /keras-yolov3
COPY . /keras-yolov3
RUN pip install -r requirements.txt
