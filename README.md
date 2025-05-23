# Implementation of MaD

## Datasets
Please follow the instructions in [this issue](https://github.com/mlfoundations/task_vectors/issues/1) to download the datasets (e.g., Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN).


## Checkpoints
Please download the checkpoints of CLIP ViT-{B/32, B/16, L/14} models at [here](https://drive.google.com/drive/folders/15ParSng4d5xSdaWdBFsg1617zPXT8Dae?usp=sharing).
```sh
/path/to/checkpoints
├── model_1 (e.g., ViT-B-32)
│   ├── dataset_1 (e.g., Cars)
│   │   ├── <your own checkpoint file name 1>.pt
│   │   ├── <your own checkpoint file name 2>.pt   
│   │   ...
│   │   └── <your own checkpoint file name N>.pt
│   ├── dataset_2
│   ...
│   ├── dataset_T
│   └── zeroshot.pt
├── model_2
...
└── model_M
```


## Dependencies
### 0. Install Poetry (w/ Python 3.8)
Please follow the instructions in https://python-poetry.org/docs/.

You can also install Poetry within a Conda environment.
```sh
conda create -n <env name> python=3.8 -y
conda activate <env name>
pip install poetry
```
### 1. Install dependencies with Poetry
```sh
poetry install
```

## Running code
```sh
poetry run python src/mad.py \
    --model ViT-B-32 \
    --num-tasks 8 \
    --n-samples 32 \
    --bank-type train \
    --tallmask-setting \
    --batch-size 32

```