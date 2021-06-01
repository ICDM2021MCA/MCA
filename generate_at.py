from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
from skimage.io import imread, imsave
from utils import NormalizeByChannelMeanStd


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', type=str, default='cuda:0',
                    help='gpu')
parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./model-cifar10-ResNet18/CAR-MADRY/car_beta-0.01-madry/model-nn-test.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./model-cifar10-ResNet18/VANILLA/vanilla/model-nn-epoch100.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./model-cifar10-ResNet18/CAR-TRADES/car_beta-0.05-trades_beta-6.0/model-nn-test.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--distance', type=str, default='l_inf', help='l_inf, l_1, l_2 ball')


args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(args.gpu if use_cuda else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

standardize = NormalizeByChannelMeanStd(
    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]).to(device)

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def _pgd_whitebox(model,
                  x_natural,
                  y,
                  epsilon=args.epsilon,
                  perturb_steps=args.num_steps,
                  step_size=args.step_size,
                  distance=args.distance):

    batch_size = len(x_natural)
    norm = distance.split('_')[1]

    if args.random:
        delta = torch.rand_like(x_natural) * 2 * epsilon - epsilon
        if distance != 'l_inf':  # projected into feasible set if needed
            normVal = torch.norm(delta.view(batch_size, -1), int(norm), 1)
            normVal = torch.max(normVal, torch.ones_like(norm) * 1e-6)
            mask = normVal <= epsilon
            scaling = epsilon / normVal
            scaling[mask] = 1
            delta = delta * scaling.view(batch_size, 1, 1, 1)
    else:
        delta = torch.zeros_like(x_natural)
    x_adv = x_natural.detach() + delta.detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            _, logit_adv = model(x_adv)
            loss = nn.CrossEntropyLoss()(logit_adv, y)
        updates = torch.autograd.grad(loss, [x_adv])[0]
        if distance == 'l_inf':
            updates = torch.sign(updates)
        else:
            normVal = torch.norm(updates.view(batch_size, -1), int(norm), 1)
            normVal = torch.max(normVal, torch.ones_like(norm) * 1e-6)
            updates = updates / normVal.view(batch_size, 1, 1, 1)

        updates = updates * step_size
        x_adv = x_adv.detach() + updates.detach()

        # projection
        delta = x_adv - x_natural
        if distance == 'l_inf':
            delta = torch.clamp(delta, -epsilon, epsilon)
        else:
            normVal = torch.norm(delta.view(batch_size, -1), int(norm), 1)
            normVal = torch.max(normVal, torch.ones_like(norm) * 1e-6)
            mask = normVal <= epsilon
            scaling = epsilon / normVal
            scaling[mask] = 1
            delta = delta * scaling.view(batch_size, 1, 1, 1)

        x_adv = torch.clamp(x_natural + delta, 0.0, 1.0)

    return x_adv


def bhwc2bchw(x):
    if isinstance(x, np.ndarray):
        pass
    else:
        raise

    if x.ndim == 3:
        return np.moveaxis(x, 2, 0)
    if x.ndim == 4:
        return np.moveaxis(x, 3, 1)


def bchw2bhwc(x):
    if isinstance(x, np.ndarray):
        pass
    else:
        raise

    if x.ndim == 3:
        return np.moveaxis(x, 0, 2)
    if x.ndim == 4:
        return np.moveaxis(x, 1, 3)


def main():

    img_file = './images/bird8.png'
    np_img = np.array(imread(img_file)/255)
    img = torch.tensor(bhwc2bchw(np_img))[None, :, :, :].float().to(device)
    label = torch.tensor(np.array([2])).to(device)

    model = ResNet18().to(device)
    model = nn.Sequential(standardize, model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    img_adv = bchw2bhwc(np.squeeze(_pgd_whitebox(model, img, label).cpu().numpy()))
    imsave('./images/bird8_adv.png', img_adv)


if __name__ == '__main__':
    main()
