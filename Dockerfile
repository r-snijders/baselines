FROM ubuntu:16.04

RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake
RUN apt-get -y install swig
ENV CODE_DIR /root/code
ENV VENV /root/venv

COPY . $CODE_DIR/baselines
RUN \
    pip install virtualenv && \
    virtualenv $VENV --python=python3 && \
    . $VENV/bin/activate && \
    cd $CODE_DIR && \
    pip install --upgrade pip && \
    pip install -e baselines && \
    pip install pytest Box2D matplotlib

ENV PATH=$VENV/bin:$PATH
WORKDIR $CODE_DIR/baselines

#Used for atari environments:
RUN apt-get -y install libglib2.0-0 libsm6 libxrender1
#Used for cartpole environments:
RUN apt-get -y install libglib2.0-0 python-opengl

RUN apt-get -y install python3-tk

#Vim:
RUN apt-get -y install vim
COPY .vimrc /root

WORKDIR /root

CMD /bin/bash
