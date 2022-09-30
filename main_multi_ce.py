from __future__ import print_function
import argparse
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from triplet_image_loader import TripletImageLoader
from tripletnet import CS_Tripletnet
from data_loader import SimpleDataManager
from supcon import SupConLoss, CondSupConLoss
import numpy as np
import Resnet_18
from ccn import ConditionalContrastiveNetwork, MultiTaskNetwork

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MultiTask Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum',
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='weight_decay',
                    help='weight_decay')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Multi_Task_Network', type=str,
                    help='name of experiment')
parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for mask norm')
parser.add_argument('--dim_embed', type=int, default=128, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--metadata', type=str, default='/data/ddmg/xray_data/zappos50k_data/zap50k_meta.csv',
                    help='metadata filename')
parser.set_defaults(test=False)
parser.set_defaults(visdom=False)

best_acc = 0


def main():
    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("Loading Data ...")
    train_datamgr = SimpleDataManager(112, batch_size=args.batch_size, supcon=False)
    train_loader = train_datamgr.get_data_loader(args.metadata, split='train', aug=True)

    print("Setting up Model ...")
    model = Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)
    mt_model = MultiTaskNetwork(model, embedding_size=args.dim_embed, cond_tasks=[4,5,4])

    if args.cuda:
        mt_model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            ccn_model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mt_model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, mt_model, criterion, optimizer, epoch, args.cuda)

        if epoch % 50 == 0:
            # remember best acc and save checkpoint
            directory = "runs/%s/" % (args.name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = directory + "lr" + str(args.lr) + "_epoch" + str(epoch) + '_checkpoint.pth.tar'
            torch.save({'epoch': epoch, 'state': mt_model.state_dict()}, filename)


def train(train_loader, model, criterion, optimizer, epoch, cuda):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.float()

        if cuda:
            images = images.cuda(non_blocking=True)
            labels = [l.cuda() for l in labels]
        bsz = labels[0].shape[0]

        features = model(images)
        loss = 0
        for i in range(len(features)):
            loss += criterion(features[i], labels[i])

        # update metric
        losses.update(loss.item(), bsz)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} {}\t'
                  'Loss: {:.4f} ({:.4f}) \t'.format(
                epoch, batch_idx,
                losses.val, losses.avg))


def save_checkpoint(state, epoch, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'epoch_' + epoch + filename
    torch.save(state, filename)


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name],
                                 name=split_name)

    def plot_mask(self, masks, epoch):
        self.viz.bar(
            X=masks,
            env=self.env,
            opts=dict(
                stacked=True,
                title=epoch,
            )
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((1 - 0.015) ** epoch)
    if args.visdom:
        plotter.plot('lr', 'learning rate', epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum() * 1.0 / dista.size()[0]


def accuracy_id(dista, distb, c, c_id):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return ((pred > 0) * (c.cpu().data == c_id)).sum() * 1.0 / (c.cpu().data == c_id).sum()


if __name__ == '__main__':
    main()    
