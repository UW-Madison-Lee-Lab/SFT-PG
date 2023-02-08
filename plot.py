from data import get_dataset, fix_legacy_dict
from torch.utils.data import DataLoader
import numpy as np
import os
from torchvision.utils import save_image
import torch
import cv2
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '--name', type=str, default = 'cifar10', choices=["cifar10", "celeba64"], help='Name of experiment')
    parser.add_argument('-dataset-path', '--dataset-path', type=str, default = './data')
    parser.add_argument('-data-path', '--data-path', type=str, default = './generated')
    parser.add_argument('-n', '--n_generate', type=int, help='Number of samples to generate', default = 50048)
    args = parser.parse_args()

    train_set = get_dataset(args.name, args.dataset_path)
    train_loader = DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    for step, (images, labels) in enumerate(train_loader):

        assert (images.max().item() <= 1) and (0 <= images.min().item())
        if images_list.shape[0] == 0:
            images_list = images.cpu().clone().numpy()
        else: 
            # print(step)
            images_list = np.concatenate([images_list, images.cpu().clone().numpy()])
        if images_list.shape[0] >= args.n_generate:
            break

    # print(images_list.shape)
    images_list = torch.tensor(images_list)
    for index in range(images_list.shape[0]):
        save_image(images_list[index], fp=os.path.join(args.data_path, args.name, '{}.jpg'.format(index)))