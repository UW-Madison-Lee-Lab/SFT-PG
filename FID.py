import os
import torch
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # # dataset and model
    # parser.add_argument('-name', '--name', type=str, default = 'cifar10', choices=["cifar10", "lsun_bedroom", "celeba64"],
    #                     help='Name of experiment')
    # parser.add_argument('-ema', '--ema', action='store_true', default = True, help='Whether use ema')

    # # fast generation parameters
    # parser.add_argument('-approxdiff', '--approxdiff', type=str, default = 'VAR', choices=['STD', 'STEP', 'VAR'], help='approximate diffusion process')
    # parser.add_argument('-kappa', '--kappa', type=float, default=1.0, help='factor to be multiplied to sigma')
    # parser.add_argument('-S', '--S', type=int, default=20, help='number of steps')
    # parser.add_argument('-schedule', '--schedule', type=str, default = 'quadratic', choices=['linear', 'quadratic'], help='noise level schedules')

    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('-generated_path', '--generated_path', type = str, default = './generated/ema_celeba64_VAR10quadratic_kappa1.0finetune_generate')
    parser.add_argument('-data_path', '--data_path', type = str, default = './generated/celeba')

    args = parser.parse_args()
    
    kwargs = {'batch_size': 50, 'device': torch.device('cuda:{}'.format(args.gpu)), 'dims': 2048}
    paths = [args.generated_path, args.data_path]
    # './generated/{}'.format(folder)]
    fid = calculate_fid_given_paths(paths=paths, **kwargs)
    print('FID = {}'.format(fid))