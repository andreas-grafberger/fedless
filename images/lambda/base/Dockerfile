FROM public.ecr.aws/lambda/python:3.8

ENV TF_ENABLE_ONEDNN_OPTS=1

# Install tensorflow beforehand to improve docker build/push times
RUN pip install tensorflow==2.5

RUN yum install -y git

# Install fedless from built wheel
COPY ./dist/fedless*.whl .
RUN pip install fedless*.whl