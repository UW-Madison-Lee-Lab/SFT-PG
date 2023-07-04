# Toy experiments

## Pre-training
```
cd ./toy_exp
python train.py --pretrain True
```
It would save the pretrained model as `./toy_exp/swiss.pt`.

## Fine-tuning 
```
python train.py --pretrain False
```
It would fine-tuned the model initialized with `./toy_exp/swiss.pt`.

The diffusion model part of the code is adapted from [https://github.com/acids-ircam/diffusion_models](https://github.com/acids-ircam/diffusion_models).
