from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.small_cnn import *
from models.net_mnist import *


parser = argparse.ArgumentParser(description='PyTorch MNIST PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.3,
                    help='perturbation')
parser.add_argument('--num-steps', default=40,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.01,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./model-mnist-smallCNN/MADRY/madry/model-nn-test.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./model-mnist-smallCNN/MADRY/madry/model-nn-epoch100.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./model-mnist-smallCNN/CAR-TRADES/car_beta-0.05-trades_beta-6.0/model-nn-test.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--distance', type=str, default='l_inf', help='l_inf, l_1, l_2 ball')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def _pgd_whitebox(model,
                  x_natural,
                  y,
                  epsilon=args.epsilon,
                  perturb_steps=args.num_steps,
                  step_size=args.step_size,
                  distance=args.distance):
    _, out = model(x_natural)
    err = (out.data.max(1)[1] != y.data).float().sum()
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

    _, logit_adv = model(x_adv)
    err_pgd = (logit_adv.data.max(1)[1] != y.data).float().sum()
    print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  x_natural,
                  y,
                  epsilon=args.epsilon,
                  perturb_steps=args.num_steps,
                  step_size=args.step_size,
                  distance=args.distance):
    _, out = model_target(x_natural)
    err = (out.data.max(1)[1] != y.data).float().sum()
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
    x_adv = x_natural + delta

    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            _, logit_adv = model_source(x_adv)
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

    _, logit_adv = model_target(x_adv)
    err_pgd = (logit_adv.data.max(1)[1] != y.data).float().sum()
    print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('natural_acc_total: ', 1 - (natural_err_total / len(test_loader.dataset)))
    print('robust_err_total: ', robust_err_total)
    print('robust_acc_total: ', 1 - (robust_err_total / len(test_loader.dataset)))


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('natural_acc_total: ', 1 - (natural_err_total / len(test_loader.dataset)))
    print('robust_err_total: ', robust_err_total)
    print('robust_acc_total: ', 1 - (robust_err_total / len(test_loader.dataset)))


def main():

    if args.white_box_attack == True:
        # white-box attack
        print('pgd white-box attack')
        model = SmallCNN().to(device)
        model.load_state_dict(torch.load(args.model_path))

        eval_adv_test_whitebox(model, device, test_loader)
    else:
        # black-box attack
        print('pgd black-box attack')
        model_target = SmallCNN().to(device)
        model_target.load_state_dict(torch.load(args.target_model_path))
        model_source = SmallCNN().to(device)
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    main()
