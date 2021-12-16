FROM python:3.8

ENV TF_ENABLE_ONEDNN_OPTS=1

RUN pip install flwr==0.16.0
RUN pip install tensorflow==2.5

COPY ./dist/fedless*.whl .
RUN pip install fedless*.whl
