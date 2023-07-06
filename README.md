# SFT-PG

Code for [Optimizing DDPM Sampling with Shortcut Fine-Tuning](https://arxiv.org/abs/2301.13362), ICML 2023. 

## Requirements
See requirements.txt. Our experiments are conducted on Ubuntu Linux 20.04 with Python 3.8.

## Datasets and pre-trained models
Download pretrained models from [here](https://drive.google.com/file/d/1EuoxEVJwRhHWfYV4bhZWjVPyW6cS90Y-/view?usp=sharing) and unzip as `./checkpoints`.

Download datasets from [here](https://drive.google.com/file/d/1fq0yQZS-jCcuYeMBXu-YXKmVL7QByYMb/view?usp=sharing) and unzip as `./data`.

## Fine-tuning
For CIFAR 10:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8106 finetune.py --name cifar10 --img_shape 32
```

For CelebA:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8106 finetune.py --name celeba64 --img_shape 64
```

The default batch size is 128 and can be adjusted by `--batchsize` no matter how many gpus are used in distributed training.

## Generating images

```
python generate.py --name dataset_name
```
where `dataset_name` is either `cifar10` or `celeba64`.
## Computing FID

```
python FID.py --name dataset_name --data_path path_to_dataset
```
where `dataset_name` is either `cifar10` or `celeba64`, and `data_path` is the path to the folder of ground truth images.

Some parts of the sampling code are adapted from [FastDPM](https://github.com/FengNiMa/FastDPM_pytorch), where we use FastDPM with the pretrained model as initialization. 

## For toy dataset
Please check `./toy_exp` for details.


##
If you find the code useful, please cite:
```
@article{fan2023optimizing,
  title={Optimizing DDPM Sampling with Shortcut Fine-Tuning},
  author={Fan, Ying and Lee, Kangwook},
  journal={arXiv preprint arXiv:2301.13362},
  year={2023}
}
```
