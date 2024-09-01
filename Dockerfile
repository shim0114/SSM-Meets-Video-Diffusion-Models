# nvidia docker image
# Select base image: https://catalog.ngc.nvidia.com/containers
# https://qiita.com/k_ikasumipowder/items/32bf0bc781cbbdfa2edb#%E8%A4%87%E6%95%B0%E3%81%AE%E3%83%90%E3%83%BC%E3%82%B8%E3%83%A7%E3%83%B3%E3%81%AE%E3%82%B3%E3%83%B3%E3%83%86%E3%83%8A%E3%81%AE%E3%83%86%E3%82%B9%E3%83%88
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && apt-get install -y sudo python3 python3-pip git

RUN pip3 install --upgrade pip
# # requirements.txtをコンテナにコピー
# COPY requirements.txt /tmp/requirements.txt
# # pip install で必要なパッケージをインストール
# RUN pip3 install -r /tmp/requirements.txt

# Pythonライブラリのインストール
RUN pip3 install numpy pandas matplotlib scikit-learn jupyterlab 
RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install wandb einops_exts rotary_embedding_torch av 
RUN pip3 install imageio minerl_navigate tensorflow_gan tensorflow-hub==0.15.0
# https://keep-loving-python.hatenablog.com/entry/2022/03/12/144617
RUN pip3 install tensorflow-probability==0.23.0
RUN pip3 install -q -U einops datasets 
# for s6
RUN pip3 install mamba-ssm
# # for t2v
# RUN pip install pip install transformers -U
# RUN pip install pip install sentencepiece sacremoses importlib_metadata
# for squid 
RUN pip3 install requests==2.24

########################################################################################
# # 基本イメージ
# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# ENV DEBIAN_FRONTEND noninteractive

# RUN apt-get update -y && apt-get -y upgrade && apt-get install -y \
#     wget curl git libgl1-mesa-dev 

# # Pythonライブラリのインストール
# # RUN pip install numpy pandas matplotlib scikit-learn jupyterlab 
# # RUN pip install torch torchvision
# RUN pip install jaxlib wandb einops_exts rotary_embedding_torch av 
# RUN pip install imageio minerl_navigate tensorflow_gan tensorflow-hub==0.15.0
# RUN pip install tensorflow-probability==0.23.0
# RUN pip install -q -U einops datasets 
# # for s6
# RUN pip install mamba-ssm
# # for t2v
# RUN pip install pip install transformers -U
# RUN pip install pip install sentencepiece sacremoses importlib_metadata
# # for squid 
# RUN pip install requests==2.24 

# RUN apt-get install -y unrar-free ffmpeg
# RUN apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# WORKDIR /workspace 

# # bash シェルをデフォルトコマンドとして設定
# CMD ["bash"]


########################################################################################
# # S6 以前

# # 基本イメージとしてPython 3のオフィシャルイメージを使用
# FROM python:3.10

# ENV DEBIAN_FRONTEND noninteractive

# RUN apt-get update -y && apt-get -y upgrade && apt-get install -y \
#     wget curl git libgl1-mesa-dev

# # Pythonライブラリのインストール
# RUN pip install numpy pandas matplotlib scikit-learn jupyterlab 
# RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# RUN pip install wandb einops_exts rotary_embedding_torch av 
# RUN pip install imageio minerl_navigate tensorflow_gan tensorflow-hub==0.15.0
# # https://keep-loving-python.hatenablog.com/entry/2022/03/12/144617
# RUN pip install tensorflow-probability==0.23.0
# RUN pip install -q -U einops datasets 
# # for s6
# RUN pip install mamba-ssm
# # for t2v
# RUN pip install pip install transformers -U
# RUN pip install pip install sentencepiece sacremoses importlib_metadata
# # for squid 
# RUN pip install requests==2.24

# RUN apt-get install -y unrar-free ffmpeg
# RUN apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# WORKDIR /workspace 

# # bash シェルをデフォルトコマンドとして設定
# CMD ["bash"]
