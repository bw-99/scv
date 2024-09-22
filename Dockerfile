
# CUDA 12.1과 Python 3.9이 포함된 NVIDIA의 공식 베이스 이미지 사용
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04


# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive

# 패키지 업데이트 및 tzdata 설치
RUN apt-get update && \
    apt-get install -y tzdata

# 서울 시간대 설정
RUN ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata


# Python 3.9 설치
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils && \
    apt-get install -y wget curl && \
    ln -fs /usr/bin/python3.9 /usr/bin/python3 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# Set the working directory in the container
WORKDIR /app

# Install necessary dependencies
RUN pip install --upgrade pip

RUN apt update && \
    apt install -y git

# Install the required Python packages
RUN pip install \
    torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html \
    pandas==1.5.3 \
    numpy==1.26.4 \
    scipy==1.13.0 \
    scikit-learn==1.3.0 \
    pyyaml==6.0 \
    h5py==3.8.0 \
    tqdm==4.66.1 \
    keras_preprocessing==1.1.2 \
    polars==1.1.0 \
    pyarrow==16.1.0 \
    git+https://github.com/reczoo/FuxiCTR.git

# Set default command to run Python
# CMD ["/bin/bash"]
ENTRYPOINT [ "/bin/bash" ]
