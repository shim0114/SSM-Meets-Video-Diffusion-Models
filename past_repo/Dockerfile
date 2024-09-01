# 基本イメージとしてPython 3のオフィシャルイメージを使用
FROM python:3.10

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && apt-get -y upgrade && apt-get install -y \
    wget curl git libgl1-mesa-dev

# Pythonライブラリのインストール
RUN pip install numpy pandas matplotlib scikit-learn jupyterlab 
RUN pip install torch==2.0.1 torchvision==0.15.2
RUN pip install s5-pytorch jaxlib wandb einops_exts rotary_embedding_torch av 
RUN pip install imageio minerl_navigate tensorflow tensorflow_gan tensorflow-hub==0.15.0
RUN pip install -q -U einops datasets 
RUN pip install requests==2.24 

RUN apt-get install -y unrar-free ffmpeg
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace 

# bash シェルをデフォルトコマンドとして設定
CMD ["bash"]
