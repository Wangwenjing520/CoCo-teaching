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
from torchvision.models import vgg19_bn


import time
import argparse
from common.utils import sigmoid_rampup
from algorithm.loss import loss_coco_no_pure_ratio

from torch.autograd import Variable
from data import animal10n

import wandb



wandb.login()

import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser(description='PyTorch Clothing-1M resnet50 Training')
parser.add_argument('--lr', default=0.02, type=float, help='learning_rate')
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=7)
parser.add_argument('--use_flip', default=False, type=bool)
parser.add_argument('--use_drop', default=False, type=bool)  # 是否使用丢弃率
parser.add_argument('--use_dynamic', default=True, type=bool)  # 是否使用动态
parser.add_argument('--alpha', default=1, type=float)  # 是否使用动态
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

    if epoch < 50:
        learning_rate = args.lr
    elif 50 <= epoch < 100:
        learning_rate = 0.002
    else:
        learning_rate = 0.0002
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    if args.use_drop:  # 如果丢弃样本
        forget_rate = 0.08
        rate_schedule = np.ones(args.num_epochs) * forget_rate
        rate_schedule[:15] = np.linspace(0, forget_rate, 15)  # todo:修改

    print('\n=> Animal Training Epoch #%d, LR=%.4f' % (epoch, learning_rate))
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
        # if (batch_idx+1) % 1000 == 0:
        #     val(epoch)
        #     net.train()
        #     net2.train()
    return 100. * correct / total, train_loss / len(train_loader)



def test():
    # global test_acc
    #     test_net.eval()
    with torch.no_grad():
        net.eval()
        net2.eval()

        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            outputs2 = net2(inputs)
            outputs += outputs2

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        acc = 100. * correct / total
        test_acc = acc
        print('test acc:', test_acc)
        record.write('Test Acc: %f\n' % acc)
        test_loss = test_loss / len(test_loader)
    return test_acc, test_loss


record = open('./checkpoint/' + 'animal_test.txt', 'w')
record.write('learning rate: %f\n' % args.lr)
record.flush()

loader = animal10n.animal_dataloader(batch_size=args.batch_size, num_workers=5)
train_loader, test_loader = loader.run()


# Model
print('\nModel setup')
print('| Building net')

net = vgg19_bn(pretrained=False)
net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 10)
        )

net2 = vgg19_bn(pretrained=False)
net2.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 10)
        )


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0, 0.005)
        nn.init.constant_(m.bias, 0)

net.classifier.apply(init_weights)
net2.classifier.apply(init_weights)

device = torch.device('cuda')
net = net.to(device)
net2 = net2.to(device)
net = torch.nn.DataParallel(net)
net2 = torch.nn.DataParallel(net2)
cudnn.benchmark = True

optimizer = optim.SGD(list(net.parameters()) + list(net2.parameters()), lr=args.lr,
                      momentum=0.9, weight_decay=0.0005)

criterion = nn.CrossEntropyLoss()


print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))

# global logger
logger = wandb.init(project='animal_coco',
                    name=f'{args.alpha}',
                    save_code=True)
logger.config.update(args)

args.best_acc = -1
for epoch in range(args.num_epochs):
    train_acc, train_loss = train(epoch)
    test_acc, test_loss = test()

    if test_acc > args.best_acc:
        args.best_acc = test_acc

    logger.log({'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss,
                'test_acc': test_acc, 'test_loss': test_loss,
                'best_acc': args.best_acc})

print('\nTesting model')


print('* Test results : Acc@1 = %.2f%%' % test_acc)
record.write('Test Acc: %.2f\n' % test_acc)
record.flush()
record.close()
