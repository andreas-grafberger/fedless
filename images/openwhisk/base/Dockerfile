FROM openwhisk/actionloop-python-v3.7

ENV TF_ENABLE_ONEDNN_OPTS=1

# Install tensorflow beforehand to improve docker build/push times
RUN pip install tensorflow==2.5

# Install fedless from built wheel
COPY ./dist/fedless*.whl .
RUN pip install fedless*.whl

