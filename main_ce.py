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
import time
import numpy as np
import Resnet_18
from Resnet_18 import LinearClassifier
from ccn import ConditionalContrastiveNetwork

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CCN Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--print_freq', type=int, default=10,
                    help='print frequency')


# optimization
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                    help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                    help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='weight_decay',
                    help='weight_decay')
parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum',
                    help='momentum')


# model param
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to checkpoint')
parser.add_argument('--prev_model', default='CCN', type=str,
                    help='type of previous network')
parser.add_argument('--name', default='Cross_Entropy_Network', type=str,
                    help='name of experiment')
parser.add_argument('--dim_embed', type=int, default=128, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--n_classes', type=int, default=4, metavar='N',
                    help='how many classes')
parser.add_argument('--n_classes_prev', type=int, default=4, metavar='N',
                    help='how many classes')
parser.add_argument('--dim_proj', type=int, default=32, metavar='N',
                    help='how many dimensions in projection for CCN')
parser.add_argument('--category', type=str,
                    help='category to train for')

# other
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables no CUDA training')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--warm', action='store_true',
                    help='warm-up for large batch training')
parser.set_defaults(test=False)

best_acc = 0


def main():
    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if args.warm:
        args.warm_epochs = 10
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    if args.checkpoint:
        args.name = args.name + '_' + args.prev_model
    if args.category:
        args.name = args.name + '_' + args.category

    metadata_file = 'data/zap50k_meta.csv'
    if args.category == 'Brand':
        metadata_file = 'data/zap50k_meta_brand.csv'

    print("Loading Data ...")
    train_datamgr = SimpleDataManager(112, batch_size=args.batch_size, targets=[args.category], supcon=False)
    train_loader = train_datamgr.get_data_loader(metadata_file, split='train', aug=True)

    val_datamgr = SimpleDataManager(112, batch_size=args.batch_size, targets=[args.category], supcon=False)
    val_loader = val_datamgr.get_data_loader(metadata_file, split='val', aug=False)

    test_datamgr = SimpleDataManager(112, batch_size=512, targets=[args.category], supcon=False)
    test_loader = test_datamgr.get_data_loader(metadata_file, split='test', aug=False)

    if args.checkpoint:
        print("Loading previous checkpoint ...")
        if args.prev_model == 'CCN':
            model = Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)
            ccn_model = ConditionalContrastiveNetwork(model, n_conditions=3, embedding_size=args.dim_embed,
                                                      projection_size=args.dim_proj)
            checkpoint = torch.load(args.checkpoint)
            ccn_model.load_state_dict(checkpoint['state'])
            model = ccn_model.embedding_net
        elif args.prev_model == 'CSN':
            model = Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)
            csn_model = ConditionalSimNet(model, n_conditions=len(conditions),
                                          embedding_size=args.dim_embed, learnedmask=True, prein=False)
            tnet = CS_Tripletnet(csn_model)
            checkpoint = torch.load(args.checkpoint)
            tnet.load_state_dict(checkpoint['state_dict'])
            model = tnet.embeddingnet.embeddingnet
        elif 'CE' in args.prev_model:
            model = Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)
            lin_model = LinearClassifier(model, embedding_size=args.dim_embed, n_classes=args.n_classes_prev)
            checkpoint = torch.load(args.checkpoint)
            lin_model.load_state_dict(checkpoint['state'])
            model = lin_model.embedding_net
        else:
            return ModuleNotFoundError("Model type is not found.")
        lin_model = LinearClassifier(model, embedding_size=args.dim_embed, n_classes=args.n_classes)
        for param in lin_model.embedding_net.parameters():
            param.requires_grad = False
        print(lin_model)
    else:
        print("Setting up Model ...")
        model = Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)
        lin_model = LinearClassifier(model, embedding_size=args.dim_embed, n_classes=args.n_classes)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(lin_model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.cuda:
        lin_model.cuda()
        criterion.cuda()

    cudnn.benchmark = True
    top1_best = 0
    test_acc = None
    test_std = None
    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, lin_model, criterion, optimizer, epoch, args)
        losses, top1, top1_std = validate(val_loader, lin_model, criterion, args)
        if top1 > top1_best:
            top1_best = top1
            # remember best acc and save checkpoint
            directory = "runs/%s/" % (args.name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = directory + "lr" + str(args.lr) + '_checkpoint.pth.tar'
            torch.save({'epoch': epoch, 'state': lin_model.state_dict()}, filename)
            loss, test_acc, test_std = validate(test_loader, lin_model, criterion, args)

    print("Best Test Acc:")
    print(test_acc)
    print("Best Test Std:")
    print(test_std)


def train(train_loader, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    for idx, (features, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)

        features = features.float()

        if opt.cuda:
            features = features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = classifier(features)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, topk=(1, ))
        top1.update(acc1[0].cpu().detach().numpy()[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, classifier, criterion, opt):
    """validation"""
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_top1 = []

    with torch.no_grad():
        end = time.time()
        for idx, (features, labels) in enumerate(val_loader):
            features = features.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(features)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels)
            top1.update(acc1[0].cpu().detach().numpy()[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            all_top1.append(acc1[0].cpu().detach().numpy()[0])

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg, np.std(np.array(all_top1))


def save_checkpoint(state, epoch, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'epoch_' + epoch + filename
    torch.save(state, filename)


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


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((1 - 0.015) ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()    
