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
from data_loader import SimpleDataManager, TransformLoader, TwoCropTransform
from supcon import SupConLoss, CondSupConLoss, CondSupConLossWeighted
import numpy as np
import Resnet_18
from resnet import resnet34, resnet50
from ccn import ConditionalContrastiveNetwork, ContrastiveNetwork
import model_utils
import medic_dataset

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CCN Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
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
parser.add_argument('--name', default='Conditional_Contrastive_Network', type=str,
                    help='name of experiment')
parser.add_argument('--im_dim', type=int, default=112, metavar='N',
                    help='image dimensions in transform')
parser.add_argument('--dim_embed', type=int, default=128, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--dim_proj', type=int, default=32, metavar='N',
                    help='how many dimensions in projection')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--category', type=str, default=None,
                    help='SupCon if single category')
parser.add_argument('--simclr', action='store_true',
                    help='Run SimCLR on images')
parser.add_argument('--weights', action='store_true',
                    help='Learn weights for tasks')
parser.add_argument('--temperature', type=float, default=0.07,
                    help='SupCon temperature for task')
parser.add_argument('--metadata', type=str, default='/data/ddmg/xray_data/zappos50k_data/zap50k_meta.csv',
                    help='metadata filename')
parser.add_argument('--dataset', type=str, default='zappos50k',
                    help='dataset name')
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

    if args.dataset == 'zappos50k':
        args.data_path = '/data/ddmg/xray_data/zappos50k_data/ut-zap50k-images/'
        args.model = 'resnet18'
        args.im_dim = 112
        args.aug = True
        args.default_cat = ['Category']
        args.pacs = False
    elif args.dataset == 'trifeature':
        args.data_path = '/data/ddmg/xray_data/trifeature/color_texture_shape_stimuli/'
        args.model = 'resnet18'
        args.im_dim = 112
        args.batch_size = 64
        args.aug = False
        args.default_cat = ['shape_t']
        args.pacs = False
    elif args.dataset == 'medic':
        args.data_path = '/data/ddmg/xray_data/MEDIC/'
        args.default_cat = ['damage_severity']
        args.transform = TransformLoader(224).get_composed_transform(True, False, False)
        args.transform = TwoCropTransform(args.transform)
        args.sep = '\t'
        args.metadata = '/data/ddmg/xray_data/MEDIC/MEDIC_train.tsv'
        args.model = 'resnet50'
        args.dim_proj = 64
        args.pacs = False
    elif args.dataset == 'fitzpatrick17k':
        args.metadata = '/data/ddmg/xray_data/fitzpatrick17k/fitzpatrick17k_meta.csv'
        args.model = 'vgg16'
        args.aug = True
        args.data_path = '/data/ddmg/xray_data/fitzpatrick17k/'
        args.im_dim = 224
        args.dim_proj = 64
        args.default_cat = ['fitzpatrick_v']
        args.pacs = False
    elif args.dataset == 'pacs':
        import deeplake
        train_ds = deeplake.load("hub://activeloop/pacs-train")
        args.model = 'resnet50'
        args.transform = TransformLoader(224).get_composed_transform(True, False, True)
        args.transform = TwoCropTransform(args.transform)
        args.dim_proj = 64
        args.pacs = True
    else:
        raise ValueError('Dataset not defined.')

    if args.category:
        args.category = args.category.split(',')
        if len(args.category) == 1:
            args.name = args.name + '_' + args.category[0]

    print(args.category)

    if args.category:
        n_conditions = len(args.category)
    else:
        n_conditions = 0

    if not args.category:
        args.category = args.default_cat

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    print("Loading Data ...")
    if args.dataset in ['zappos50k', 'trifeature', 'fitzpatrick17k']:
        train_datamgr = SimpleDataManager(args.im_dim, batch_size=args.batch_size, targets=args.category,
                                          data_path=args.data_path, supcon=True)
        train_loader = train_datamgr.get_data_loader(args.metadata, split='train', aug=args.aug)
    elif args.dataset == 'medic':
        train_dataset = medic_dataset.MultitaskDataset(args.metadata, args.sep, args.data_path, args.transform,
                                                       args.category)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=12)
    elif args.dataset == 'pacs':
        if args.category == 'labels':
            train_loader = train_ds.pytorch(num_workers=12, batch_size=args.batch_size,
                                            transform={'images': args.transform, 'labels': None}, shuffle=True)
        elif args.category == 'domains':
            train_loader = train_ds.pytorch(num_workers=12, batch_size=args.batch_size,
                                            transform={'images': args.transform, 'domains': None}, shuffle=True)
        else:
            train_loader = train_ds.pytorch(num_workers=12, batch_size=args.batch_size,
                                            transform={'images': args.transform, 'labels': None, 'domains': None},
                                            shuffle=True)

    print("Setting up Model ...")
    if args.model == 'resnet18':
        model = Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)
    elif args.model in ['resnet50', 'vgg16']:
        model, resize_image_size, input_image_size, args.dim_embed = model_utils.initialize_model(args.model)
    else:
        raise InputError("Model is not available.")

    print(args.dim_embed)

    if n_conditions > 1:
        ccn_model = ConditionalContrastiveNetwork(model, n_conditions=n_conditions, embedding_size=args.dim_embed,
                                                  projection_size=args.dim_proj)
    else:
        ccn_model = ContrastiveNetwork(model, embedding_size=args.dim_embed, projection_size=args.dim_proj)

    if args.cuda:
        ccn_model.cuda()

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

    if args.weights:
        assert n_conditions > 1
        criterion = CondSupConLossWeighted(model=ccn_model, n_conditions=n_conditions, temperature=args.temperature)
        if args.cuda:
            criterion.cuda()
        optimizer = torch.optim.SGD(criterion.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        if n_conditions > 1:
            criterion = CondSupConLoss(n_conditions=n_conditions)
        else:
            criterion = SupConLoss()

        optimizer = torch.optim.SGD(ccn_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        if n_conditions == 0:
            train_simclr(train_loader, ccn_model, criterion, optimizer, epoch, args.cuda, args.pacs)
        elif n_conditions == 1:
            train_supcon(train_loader, ccn_model, criterion, optimizer, epoch, args.cuda, args.pacs)
        elif n_conditions > 1:
            train_contrastive(train_loader, ccn_model, criterion, optimizer, epoch, args.cuda, args.weights, args.pacs)

        if epoch % 50 == 0:
            # remember best acc and save checkpoint
            directory = "runs/%s/" % (args.name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = directory + "lr" + str(args.lr) + "dim" + str(args.dim_proj) \
                       + "_epoch" + str(epoch) + '_checkpoint.pth.tar'
            torch.save({'epoch': epoch, 'state': ccn_model.state_dict()}, filename)


def train_simclr(train_loader, model, criterion, optimizer, epoch, cuda, pacs=False):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    for batch_idx, data in enumerate(train_loader):
        if pacs:
            images = data['images']
            labels = data['labels']
        else:
            (images, labels) = data
        images = torch.cat([images[0], images[1]], dim=0)

        if cuda:
            images = images.cuda(non_blocking=True)

        bsz = labels.shape[0]

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        all_feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(all_feat)

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


def train_supcon(train_loader, model, criterion, optimizer, epoch, cuda, pacs=False):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    for batch_idx, data in enumerate(train_loader):

        if pacs:
            images = data['images']
            if 'labels' in data.keys():
                labels = data['labels']
            elif 'domains' in data.keys():
                labels = data['domains']
        else:
            (images, labels) = data

        images = torch.cat([images[0], images[1]], dim=0)

        if cuda:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        all_feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(all_feat, labels)

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


def train_contrastive(train_loader, ccn_model, criterion, optimizer, epoch, cuda, weights, pacs=False):
    losses = AverageMeter()

    # switch to train mode
    ccn_model.train()
    for batch_idx, data in enumerate(train_loader):

        if pacs:
            images = data['images']
            labels = torch.squeeze(data['labels'])
            domains = torch.squeeze(data['domains'])
            labels = [labels, domains]
        else:
            (images, labels) = data

        images = torch.cat([images[0], images[1]], dim=0)
        labels = torch.stack(labels)

        if cuda:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels[0].shape[0]

        if weights:
            loss = criterion(images, labels)

            # update metric
            losses.update(loss.item(), bsz)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(criterion.weights)
                print('Train Epoch: {} {}\t'
                      'Loss: {:.4f} ({:.4f}) \t'.format(
                    epoch, batch_idx,
                    losses.val, losses.avg))
        else:
            features = ccn_model(images)
            all_feat = []
            for i in range(len(features)):
                f1, f2 = torch.split(features[i], [bsz, bsz], dim=0)
                all_feat.append(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1))
            loss = criterion(all_feat, labels)

            # update metric
            losses.update(loss.item(), bsz)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(criterion.weights)
                print('Train Epoch: {} {}\t'
                      'Loss: {:.4f} ({:.4f}) \t'.format(
                    epoch, batch_idx,
                    losses.val, losses.avg))


def save_checkpoint(state, epoch, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
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
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name], name=split_name)
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
    return (pred > 0).sum()*1.0/dista.size()[0]


def accuracy_id(dista, distb, c, c_id):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return ((pred > 0)*(c.cpu().data == c_id)).sum()*1.0/(c.cpu().data == c_id).sum()


if __name__ == '__main__':
    main()    
