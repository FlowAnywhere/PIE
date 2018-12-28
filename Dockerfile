#FROM ubuntu:16.04
FROM tensorflow/tensorflow:1.9.0-py3

LABEL maintainer="Ray Liu <ray.liu@toronto.ca>"

ADD . /PIE
WORKDIR /PIE

RUN apt-get update && apt-get install -y curl unzip && \
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - && \
    apt-get update && apt-get install -y tensorflow-model-server && \
    pip3 install -r requirements.txt && \
    curl https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip && unzip uncased_L-12_H-768_A-12.zip && mkdir /PIE/data/bert && mv uncased_L-12_H-768_A-12/* /PIE/data/bert && \
    rm -rf uncased_L-12_H-768_A-12 && rm uncased_L-12_H-768_A-12.zip && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/PIE

RUN chmod +x StartServing.sh
EXPOSE 19999
#CMD ["/PIE/StartServing.sh"]
