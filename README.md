# Exact Group Fairness Regularization via Classwise Robust Optimization (ICLR 2023)

This repository is the official implementation of the paper **Exact Group Fairness Regularization via Classwise Robust Optimization**. 

## Requirements
- GPU RTX A5000
- Python 3 / Pytorch 1.8 / CUDA 11.2

## Dataset
- Adult
- COMPAS
- UTKFace
- CivilComments-WILDS
- CelebA

## Training
To train the model(s) in the paper, run this command:
```
# Adult
$ python ./main.py --date 220101 --model lr --method lgdro_chi --lr 0.001 --epochs 70 --optim AdamW --img-size 224 --batch-size 128 --labelwise --record --margin --optim-q ibr_ip --trueloss --dataset adult --rho 5.0 --seed 0 --weight-decay 0.0001

# COMPAS
$ python ./main.py --date 220101 --model lr --method lgdro_chi --lr 0.001 --epochs 70 --optim AdamW --img-size 224 --batch-size 128 --labelwise --record --margin --optim-q ibr_ip --trueloss --dataset compas --rho 3.0 --seed 0 --weight-decay 0.0001

# UTKFace
$ python ./main.py --date 220101 --model resnet18 --method lgdro_chi --lr 0.001 --epochs 70 --optim AdamW --img-size 224 --batch-size 128 --labelwise --record --margin --optim-q ibr_ip --trueloss --dataset utkface --rho 0.3 --seed 0 --weight-decay 0.0001

# CivilComments-WILDS
$ python ./main.py --date 220101 --model bert --method lgdro_chi --lr 2e-05 --epochs 3 --optim AdamW --batch-size 24 --labelwise --record --margin --optim-q ibr_ip --trueloss --dataset jigsaw --rho 0.3 --seed 0 --weight-decay 0.0001 --uc

# CelebA
$ python ./main.py --date 220101 --model resnet18 --method lgdro_chi --lr 0.001 --epochs 70 --optim AdamW --img-size 224 --batch-size 128 --labelwise --record --margin --optim-q ibr_ip --trueloss --dataset celeba --rho 1.5 --seed 0 --weight-decay 0.0001 --target Blond_Hair
```
