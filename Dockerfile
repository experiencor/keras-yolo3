FROM python:3.7

WORKDIR /keras-yolo3
COPY ./requirements.txt /keras-yolo3/requirements.txt
RUN pip install -r requirements.txt
