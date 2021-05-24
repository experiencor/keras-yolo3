FROM python:3.7

WORKDIR /keras-yolov3
COPY ./requirements.txt /keras-yolov3/requirements.txt
RUN pip install -r requirements.txt
