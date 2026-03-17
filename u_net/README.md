# Hybrid U-Net v1.0 with PyTorch: 탄소 포집량 예측 모델

<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.12.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.7.13-blue.svg?logo=python&style=for-the-badge" /></a>

![수행 결과 예시](images/sample_result.png)

- [Hybrid U-Net v1.0 with PyTorch: 탄소 포집량 예측 모델](#hybrid-u-net-v10-with-pytorch-탄소-포집량-예측-모델)
  - [도커 설치](#도커-설치)
  - [모델 설명](#모델-설명)
    - [모델 구조](#모델-구조)
    - [입력](#입력)
    - [출력](#출력)
    - [성능](#성능)
  - [사용 방법](#사용-방법)
    - [Docker](#docker)
    - [학습(Training)](#학습training)
    - [추론(Inference)](#추론inference)
  - [Tensorboard](#tensorboard)
  - [사전 학습 모델](#사전-학습-모델)
    - [학습 조건](#학습-조건)
  - [데이터](#데이터)
  - [라이센스](#라이센스)

## 도커 설치

1. [Docker 버전 20.10.16 이상 설치:](https://docs.docker.com/get-docker/)

```bash
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

2. [NVIDIA container 설치:](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

3. docker 설치 확인

```bash
sudo docker run --gpus all nvidia/cuda nvidia-smi

Wed Nov 23 19:24:12 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.141.03   Driver Version: 470.141.03   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
| 66%   73C    P2   315W / 350W |  23793MiB / 24265MiB |     91%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:0E:00.0 Off |                  N/A |
| 70%   78C    P2   324W / 350W |  22879MiB / 24268MiB |     78%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1288      G   /usr/lib/xorg/Xorg                  9MiB |
|    0   N/A  N/A      1852      G   /usr/bin/gnome-shell                6MiB |
|    0   N/A  N/A    228837      C   python                          23773MiB |
|    1   N/A  N/A      1288      G   /usr/lib/xorg/Xorg                  4MiB |
|    1   N/A  N/A    228837      C   python                          22871MiB |
+-----------------------------------------------------------------------------+
```

4. [도커 이미지 다운로드 및 이미지 로드:]()

```bash
sudo docker load -i cq_dockerimage.tar
```

## 모델 설명

기존 segmentation 모델을 기반으로 quantized segmentation 예측 방법을 연구하였으나 실수의 탄소 포집량 예측 값의 정밀도 성능이 최대 class의 수(8bit 정수, 256 class)로 quantization 되어 표현되기 때문에 성능 개선에 제한이 됩니다.
이에 기존 산림수종 분류 연구에서 우수한 성능을 나타낸 U-Net을 기반으로 탄소 포집량 regression header를 추가한 새로운 방법의 hybrid U-Net 모델을 개발하였습니다.

### 모델 구조

![모델 구조](images/hybrid_unet.png)

### 입력

RGB 이미지와 임분고 이미지가 들어갑니다.

항공, 위성, NIR의 이미지 유형이 있으며, RHxRW의 크기가 위성은 256x256, 나머지 항공 및 NIR은 512x512크기로 구성되어 있습니다.

- RGB shape: (RH, RW, 3)
- 임분고 shape: (RH, RH, 2)

### 출력

모델의 결과로 식생분류에 대한 출력과 탄소 포집량에 대한 출력이 나옵니다.

또한 성능 및 GT와의 분석을 위한 visualization 이미지를 지원하여 쉽게 성능 분석이 가능합니다.

항공, 위성, NIR의 이미지 유형이 있으며, RHxRW의 크기가 위성은 256x256, 나머지 항공 및 NIR은 512x512크기로 구성되어 있습니다.

- 식생분류 shape: (RH, RW, 1)
- 탄소 포집량 shape: (RH, RH, 4)
- visualization shape: (RH*4 + 15, RH*4 + 15, 3)

### 성능

성능 지표는 상관계수 및 결정계수로 선택되었으며, 0.73, 0.50을 목표 개발되었습니다.

권역별 성능은 다음과 같습니다.

|권역||상관계수(Correlation)|결정계수(R_squared)|
|----|----|----|----|
|산림|항공(10cm,25cm)|0.882|0.671|
||항공 겨울(10cm)|0.856|0.548|
||NIR(10cm)|0.911|0.722|
||위성(10m)|0.832|0.674|
|도심|항공(10cm,25cm)|0.977|0.771|
||NIR(10cm)|0.969|0.706|

## 사용 방법

**노트: Python 3.7.13이상 사용**

[AI hub](https://www.aihub.or.kr)에서 도커 이미지 및 해당 데이터셋을 다운로드하여 사용 가능 합니다.

도커 이미지 안에는 코드가 포함되어 있습니다.

### 데이터셋 준비

코드의 datalists 디렉토리에는 학습, 검증, 추론에 사용되는 파일 목록이 csv에 포함됩니다. 이 csv는 data 디렉토리의 generate_cvs.py를 통해 생성 할 수 있습니다.

데이터셋의 디렉토리 구조를 다음과 같이 변경해 주세요. (또는 docker안의 config_mf.py 및 datalists디렉토리의 csv 파일을 수정하여도 됩니다.)

데이터셋 디렉토리명: dataset

- 산림: forest
- 도심: city
  - ┗ 원천데이터: image
    - ┗ 이미지: IMAGE
    - ┗ 임분고 이미지: SGRST_HIGH
  - ┗ 라벨링데이터: label
    - ┗탄소량: CRBN_QNTT
    - ┗수종식별: GT

### Docker

도커를 설치하지 않으신 분은 [도커 설치](#도커-설치)를 통해 먼저 도커를 설치하세요.

```bash
docker run --gpus all -v ${PWD}/outputs:/workspace/src/outputs -v ${PWD}/dataset:/workspace/dataset -it --name carbon_env cq_dockerimage/train:1.0
```

현재의 dataset 디렉토리를 통해 데이터를 volume으로 공유하여 사용하며, 결과로 생성되는 파일들이 outputs 디렉토리로 생성됩니다.

### 학습(Training)

```bash
> python3 train.py -h
usage: train.py [-h] [--net NET] [--seed SEED] [--local_rank LOCAL_RANK]
                [--tensorboard] [--num_workers NUM_WORKERS]
                [--output-dir OUTPUT_DIR]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--val_batch_size VAL_BATCH_SIZE] [--total-epoch TOTAL_EPOCH]
                [--eval-freq EVAL_FREQ] [--save-freq SAVE_FREQ]
                [--learning-rate LEARNING_RATE] [--pretrained] [--resume]
                [--opt OPT] [--lrs LRS] [--enc_dropout]
                [--image_type IMAGE_TYPE]

optional arguments:
  -h, --help            show this help message and exit
  --net NET             network name (default: UNet_carbon)
  --seed SEED           Fixed random seed (default: 0)
  --local_rank LOCAL_RANK
                        local rank for DistributedDataParallel(do not modify)
                        (default: -1)
  --tensorboard
  --num_workers NUM_WORKERS
                        num_workers (default: 0)
  --output-dir OUTPUT_DIR
                        output directory (default: None)
  --train_batch_size TRAIN_BATCH_SIZE
                        train batch size (default: 8)
  --val_batch_size VAL_BATCH_SIZE
                        validataion batch size (default: 8)
  --total-epoch TOTAL_EPOCH
                        total num epoch (default: 30000000)
  --eval-freq EVAL_FREQ
                        evaluation frequency (default: 5)
  --save-freq SAVE_FREQ
                        save frequency (default: 10)
  --learning-rate LEARNING_RATE
                        learning late (default: 0.0001)
  --pretrained          Start with pretrained model (if avail) (default:
                        False)
  --resume              resume from checkpoint (default: False)
  --opt OPT             nadam, adam (default: adam)
  --lrs LRS             cosinealr, steplr (default: cosinealr)
  --enc_dropout         dropout for encoder (default: False)
  --image_type IMAGE_TYPE
                        test image type (default: forest_SN_10)
```

권역별 학습을 위해 --image_type에 해당 권역의 코드네임을 인자로 합니다.
현재의 권역에 대한 코드네임은 forest_AP_10_25, forest_NIR_10, forest_SN_10, city_AP_10_25, city_NIR_10 등으로 되어 있습니다.

분산 GPU 학습을 지원하며 자시한 사항은 [분산 GPU 학습 가이드](https://learn.microsoft.com/ko-kr/azure/machine-learning/how-to-train-distributed-gpu)를 참고하세요

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234 train.py --eval-freq 1 --resume --image_type forest_AP_10_25
```

### 추론(Inference)

권역별 추론을 위해 --image_type에 해당 권역의 코드네임을 인자로 합니다.
각 권역별 모델 가중치는 weights 폴더에 포함하고 있습니다.

```bash
python3 inference.py --image_type forest_AP_10_25
```

## Tensorboard

TensorBoard는 머신러닝 실험을 위한 시각화 툴킷(toolkit)입니다. TensorBoard를 사용하면 손실 및 정확도와 같은 측정 항목을 추적 및 시각화하는 것, 모델 그래프를 시각화하는 것, 히스토그램을 보는 것, 이미지를 출력하는 것 등이 가능합니다. ([파이토치 한국어 튜토리얼](https://tutorials.pytorch.kr/recipes/recipes/tensorboard_with_pytorch.html#tensorboard) 참고)

해당 모델의 학습에서는 tensorboard 옵션을 통해 학습 중 손실 및 정확도를 추적하도록 지원합니다.

```bash
tensorboard --logdir ./outputs/forest_AP_10_25/tensorb
```

![tensorboard 예시](images/tensorboard.png)

## 사전 학습 모델

학습된 모델들은 각 권역에 대한 코드네임을 인자로 자동으로 선택됩니다.

### 학습 조건

학습에 사용된 하이퍼파라미터는 다음과 같습니다.

```
Epoch: 1000 
Batch size: 8
Optimizer: adam
Learning rate: 0.0001
Learning decay: cosinealr
Loss: cross entropy for segmentation, mse for regression
```

## 데이터

학습에 사용된 데이터는 AI hub에 업로드된 데이터를 사용하였으며, 각 권역별 데이터 및 학습, 검증, 테스트에 사용되는 파일의 리스트가 위치해 있습니다.

각 권역 코드네임은 아래와 같습니다.
forest_AP_10_25(산림 항공 10cm, 25cm), forest_NIR_10(산림 NIR 10cm), forest_SN_10(산림 위성 10m), city_AP_10_25(도심 항공 10cm, 25cm), city_NIR_10(도심 NIR 10cm)

## 라이센스

See the [LICENSE](LICENSE) file for license rights and limitations (MIT).
