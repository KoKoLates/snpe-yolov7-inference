FROM ubuntu:18.04

WORKDIR /data

COPY snpe-1.68.0.zip /data/
COPY android-ndk-r17c-linux-x86_64.zip /data/

RUN apt-get update && \
    apt-get install sudo && \
    apt-get install -y unzip && \
    unzip snpe-1.68.0.zip && \
    unzip android-ndk-r17c-linux-x86_64.zip && \
    mv snpe-1.68.0.3932 snpe-sdk

RUN apt-get install -y python3 python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2 && \
    python -m pip install --upgrade pip

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y python3-dev python3-matplotlib python3-numpy python3-scipy python3-skimage python3-sphinx python3-mako wget zip && \
    apt-get install -y libc++-9-dev

RUN /bin/bash -c "source /data/snpe-sdk/bin/dependencies.sh"
RUN /bin/bash -c "source /data/snpe-sdk/bin/check_python_depends.sh"

RUN apt-get install -y python3-protobuf && \
    apt-get install -y cmake && \
    apt-get install -y libprotobuf-dev protobuf-compiler

RUN pip install onnx==1.6.0

RUN echo '#!/bin/bash' >> setup.sh
RUN echo 'export ANDROID_NDK_ROOT=/data/android-ndk-r17c' > setup.sh
RUN echo 'source /data/snpe-sdk/bin/envsetup.sh -o /usr/local/lib/python3.6/dist-packages/onnx' >> setup.sh
RUN chmod +x setup.sh
RUN /bin/bash -c "source /data/setup.sh"

CMD ["/bin/bash"]