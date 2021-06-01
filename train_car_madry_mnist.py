from __future__ import print_function
import os
import argparse
import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dataset.MNIST import get_mnist_dataloaders_sample

from models.net_mnist import *
from models.small_cnn import *
from car_madry import car_madry_loss
from utils import AverageMeter
from crd.criterion import CRDLoss


parser = argparse.ArgumentParser(description='PyTorch MNIST CAR MADRY Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.3,
                    help='perturbation')
parser.add_argument('--num-steps', default=40,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.01,
                    help='perturb step size')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-mnist-smallCNN/CAR-MADRY',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--distance', type=str, default='l_inf', help='l_inf, l_1, l_2 ball')
parser.add_argument('--car_beta', type=float, default=0.1,
                    help='car coefficient')

# NCE distillation
parser.add_argument('--out_feat_dim', default=128, type=int, help='feature dimension')
parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
train_loader, test_loader, n_data = get_mnist_dataloaders_sample(batch_size=args.batch_size,
                                                                 k=args.nce_k,
                                                                 mode=args.mode)


def train(args, model, device, train_loader, optimizer, criterion_car, epoch, logger):
    losses = AverageMeter()
    losses_robust = AverageMeter()
    losses_car = AverageMeter()
    model.train()
    for batch_idx, (data, target, index, contrast_index) in enumerate(train_loader):
        data, target, index, contrast_index = data.to(device), target.to(device), index.to(device), contrast_index.to(
            device)

        optimizer.zero_grad()

        # calculate robust loss
        loss, loss_robust, loss_car = car_madry_loss(model=model,
                              x_natural=data,
                              y=target,
                              optimizer=optimizer,
                              car_beta=args.car_beta,
                              idx=index,
                              contrast_idx=contrast_index,
                              criterion_car=criterion_car,
                              step_size=args.step_size,
                              epsilon=args.epsilon,
                              perturb_steps=args.num_steps,
                              distance=args.distance)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), data.size(0))
        losses_robust.update(loss_robust.item(), data.size(0))
        losses_car.update(loss_car.item(), data.size(0))

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    logger.log_value('loss all', losses.avg, epoch)
    logger.log_value('loss robust', losses_robust.avg, epoch)
    logger.log_value('loss car', losses_car.avg, epoch)


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            train_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def _pgd_whitebox(model,
                  x_natural,
                  y,
                  epsilon=args.epsilon,
                  perturb_steps=args.num_steps,
                  step_size=args.step_size,
                  distance=args.distance):
    model.eval()
    batch_size = len(x_natural)
    norm = distance.split('_')[1]

    # random start
    delta = torch.rand_like(x_natural) * 2 * epsilon - epsilon
    if distance != 'l_inf':  # projected into feasible set if needed
        normVal = torch.norm(delta.view(batch_size, -1), int(norm), 1)
        normVal = torch.max(normVal, torch.ones_like(norm) * 1e-6)
        mask = normVal <= epsilon
        scaling = epsilon / normVal
        scaling[mask] = 1
        delta = delta * scaling.view(batch_size, 1, 1, 1)

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
    crr_pgd = (logit_adv.data.max(1)[1] == y.data).float().sum()
    return crr_pgd


def eval_test(model, device, test_loader, epoch, logger):
    model.eval()
    test_loss = 0
    correct = 0
    correct_adv = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_adv += _pgd_whitebox(model, data, target)
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Nat Accuracy: {}/{} ({:.0f}%), Adv Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        correct_adv, len(test_loader.dataset), 100. * correct_adv / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    test_adv_accuracy = correct_adv / len(test_loader.dataset)
    logger.log_value('test_natural loss', test_loss, epoch)
    logger.log_value('test_natural accuracy', test_accuracy, epoch)
    logger.log_value('test_adversarial accuracy', test_adv_accuracy, epoch)
    return test_loss, test_accuracy, test_adv_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 55:
        lr = args.lr * 0.1
    if epoch >= 75:
        lr = args.lr * 0.01
    if epoch >= 90:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_feat_dim(model):
    model.eval()
    data = torch.randn(1, 1, 28, 28).to(device)
    feat, out = model(data)
    return feat.shape[1]


def main():
    # settings
    model_dir = os.path.join(args.model_dir, 'car_beta-{}-madry'.format(args.car_beta))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    tb_folder = os.path.join('./log', model_dir)
    if not os.path.isdir(tb_folder):
        os.makedirs(tb_folder)
    logger = tb_logger.Logger(logdir=tb_folder, flush_secs=2)

    # init model, Net() can be also used here for training
    model = SmallCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    args.feat_dim = get_feat_dim(model)
    args.n_data = n_data
    criterion_car = CRDLoss(args).to(device)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, criterion_car, epoch, logger)

        # evaluation on natural examples
        print('================================================================')
        # eval_train(model, device, train_loader)
        eval_test(model, device, test_loader, epoch, logger)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-nn-epoch{}.pt'.format(epoch)))
            # torch.save(optimizer.state_dict(),
            #            os.path.join(model_dir, 'opt-nn-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
