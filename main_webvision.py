from __future__ import print_function

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

import time
import argparse

from model.InceptionResNetv2 import InceptionResNetV2
from common.utils import sigmoid_rampup, accuracy1, AverageMeter
from algorithm.loss import loss_coco_no_pure_ratio

from torch.autograd import Variable

from data.webvision import webvision_dataloader

import wandb

wandb.login()

import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser(description='PyTorch Clothing-1M resnet50 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning_rate')
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--seed', default=7)
parser.add_argument('--use_flip', default=False, type=bool)
parser.add_argument('--use_drop', default=False, type=bool)  # 是否使用丢弃率
parser.add_argument('--use_dynamic', default=True, type=bool)  # 是否使用动态
parser.add_argument('--alpha', default=0.7, type=float)  # 是否使用动态
# parser.add_argument('--id', default=1, type=float)  # 是否使用动态
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_cuda = torch.cuda.is_available()


def train(epoch):
    net.train()
    net2.train()

    train_loss = 0
    correct = 0
    total = 0

    if epoch < 51:
        learning_rate = args.lr
    elif 51 <= epoch < 101:
        learning_rate = 0.001
    else:
        learning_rate = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    if args.use_drop:  # 如果丢弃样本
        forget_rate = 0.2
        rate_schedule = np.ones(args.num_epochs) * forget_rate
        rate_schedule[:15] = np.linspace(0, forget_rate, 15)  # todo:修改


    print('\n=> Webvision Training Epoch #%d, LR=%.4f' % (epoch, learning_rate))
    for batch_idx, ([inputs, inputs1], targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)

        inputs1 = inputs1.to(device)
        inputs1 = Variable(inputs1)
        # if args.use_flip:
        #     inputs1 = transforms.RandomHorizontalFlip(p=1)(inputs)
        # else:
        #     inputs1 = inputs

        outputs = net(inputs)
        outputs2 = net2(inputs1)

        alpha = sigmoid_rampup(epoch, 15, -5.0) * args.alpha

        if args.use_drop and args.use_dynamic:
            loss = loss_coco_no_pure_ratio(outputs, outputs2, targets, rate_schedule[epoch], co_lambda=alpha)
        elif args.use_drop and not args.use_dynamic:
            loss = loss_coco_no_pure_ratio(outputs, outputs2, targets, rate_schedule[epoch], co_lambda=args.alpha)
        elif not args.use_drop and args.use_dynamic:
            loss = loss_coco_no_pure_ratio(outputs, outputs2, targets, 0, co_lambda=alpha)
        else:
            loss = loss_coco_no_pure_ratio(outputs, outputs2, targets, 0, co_lambda=args.alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch+1, args.num_epochs, batch_idx + 1, (len(train_loader.dataset) // args.batch_size) + 1,
                            loss.item(), 100. * correct / total))
        sys.stdout.flush()
        # if batch_idx % 1000 == 0:
        #     val(epoch)
        #     net.train()
        #     net2.train()
    return 100. * correct / total, train_loss / len(train_loader)


# def val(epoch):
#     global best_acc
#     with torch.no_grad():
#         net.eval()
#         net2.eval()
#
#         val_loss = 0
#         correct = 0
#         total = 0
#         for batch_idx, (inputs, targets) in enumerate(test_loader):
#             if use_cuda:
#                 inputs, targets = inputs.to(device), targets.to(device)
#             inputs, targets = Variable(inputs), Variable(targets)
#             outputs = net(inputs)
#             outputs2 = net2(inputs)
#             outputs += outputs2
#
#             loss = criterion(outputs, targets)
#             val_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += targets.size(0)
#             correct += predicted.eq(targets.data).cpu().sum()
#
#     acc = 100. * correct / total
#     print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" % (epoch, loss.item(), acc))
#     record.write('Validation Acc: %f\n' % acc)
#     record.flush()
#     if acc > best_acc:
#         best_acc = acc
#         print('| Saving Best Model ...')
#         save_point = './checkpoint/webvision.pth.tar'
#         save_checkpoint({
#             'state_dict': net.state_dict(),
#         }, save_point)


def test(loader):
    with torch.no_grad():
        net.eval()
        net2.eval()

        test_loss = 0
        total = 0
        top1 = AverageMeter()
        top5 = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            outputs2 = net2(inputs)
            outputs += outputs2

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            acc1, acc5 = accuracy1(outputs, targets, topk=(1, 5))
            top1.update(acc1[0].item(), inputs.size()[0])
            top5.update(acc5[0].item(), inputs.size()[0])

            total += targets.size(0)
        print("Testing....")
        print("| Test Epoch Acc@1: %.2f%%\tAcc@5: %.2f%%" % (top1.avg, top5.avg))
        record.write('Test Acc1: %f\n' % top1.avg)
        record.write('Test Acc5: %f\n' % top5.avg)
        test_loss = test_loss / len(test_loader)
    return top1.avg, top5.avg, test_loss


def create_model():
    model = InceptionResNetV2(num_classes=50)
    model = model.to(device)
    return model



record = open('./checkpoint/' + 'webvision_test.txt', 'w')
record.write('learning rate: %f\n' % args.lr)
record.flush()

loader = webvision_dataloader(batch_size=args.batch_size, num_workers=4)
train_loader, test_loader, test_loader2 = loader.run()

# Model
print('\nModel setup')
print('| Building net')


device = torch.device('cuda:0')
net = create_model()
net2 = create_model()
cudnn.benchmark = True

optimizer = optim.SGD(list(net.parameters()) + list(net2.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.0001)

criterion = nn.CrossEntropyLoss()


print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))


logger = wandb.init(project='webvision',
                    name=f'{args.alpha}',
                    save_code=True)
logger.config.update(args)

args.best_webvision_top1_acc = -1
args.best_imagenet_top1_acc = -1
for epoch in range(args.num_epochs):
    train_acc, train_loss = train(epoch)
    # val(epoch)
    top1, top5, test_loss = test(test_loader)
    top1_2, top5_2, test_loss2 = test(test_loader2)

    if top1 > args.best_webvision_top1_acc:
        args.best_webvision_top1_acc = top1
    if top1_2 > args.best_imagenet_top1_acc:
        args.best_imagenet_top1_acc = top1_2

    logger.log({'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss,
                'webvision top1 acc': top1, 'webvision top5 acc': top5, 'webvision test loss': test_loss,
                'imagenet top1 acc': top1_2, 'imagenet top5 acc': top5_2, 'imagenet test loss': test_loss2,
                'best webvision top1 acc:': args.best_webvision_top1_acc,
                'best imagenet top1 acc:': args.best_imagenet_top1_acc})

print('\nTesting model')


print('* Test results : Acc@1 = %.2f%%' % (top1))
record.flush()
record.close()
