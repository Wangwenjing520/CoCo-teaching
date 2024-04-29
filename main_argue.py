# -*- coding:utf-8 -*-
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from data.cifar_argue import CIFAR10, CIFAR100
import argparse, sys
import datetime
from algorithm.coco_argue import CoCo

import wandb
wandb.login()

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.8)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='cifar100')
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--num_gradual', type=int, default=15,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model_type', type=str, help='[mlp,cnn]', default='cnn')

parser.add_argument('--co_lambda', type=float, default=6)
parser.add_argument("--alpha", type=float, default=6)
parser.add_argument('--rampup_length', type=int, default=15)
parser.add_argument('--phase_shift', type=float, default=-5.0)
args = parser.parse_args()

# Seed
seed = args.seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if args.gpu is not None:
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.manual_seed(args.seed)
else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr


if args.dataset == 'cifar10':
    input_channel = 3
    num_classes = 10
    init_epoch = 20
    args.epoch_decay_start = 80
    filter_outlier = True
    args.model_type = "cnn"
    args.n_epoch = 200
    train_dataset = CIFAR10(root='/home/pris/wwj/datasets/cifar-10/',
                            download=True,
                            train=True,
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                            )

    test_dataset = CIFAR10(root='/home/pris/wwj/datasets/cifar-10/',
                           download=True,
                           train=False,
                           transform=transforms.ToTensor(),
                           noise_type=args.noise_type,
                           noise_rate=args.noise_rate
                           )

if args.dataset == 'cifar100':
    input_channel = 3
    num_classes = 100
    init_epoch = 5
    args.epoch_decay_start = 100
    args.n_epoch = 200
    filter_outlier = False
    args.model_type = "cnn"

    train_dataset = CIFAR100(root='/home/pris/wwj/datasets/cifar-100/',
                             download=True,
                             train=True,
                             transform=transforms.ToTensor(),
                             noise_type=args.noise_type,
                             noise_rate=args.noise_rate
                             )

    test_dataset = CIFAR100(root='/home/pris/wwj/datasets/cifar-100/',
                            download=True,
                            train=False,
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                            )


if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate


def main():
    # wandb
    global logger
    logger = wandb.init(project=f'{args.dataset}',
                        name=f'{args.noise_type}_{args.noise_rate}_{args.alpha}_{args.seed}',
                        save_code=True)
    logger.config.update(args)

    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')

    model = CoCo(args, train_dataset, device, input_channel, num_classes)

    epoch = 0

    # evaluate models with random weights
    test_acc1, test_acc2, test_acc, test_loss = model.evaluate(test_loader)
    with open(f'./results/ablation_study/{args.noise_type}_{args.noise_rate}_{args.name}' + '.txt', 'a') as f:
        f.write(str(epoch) + '_' + str(format(test_acc, '.2f')) + '\n')

    print(
        'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Model %.4f %%' % (
            epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, test_acc))

    args.best_acc = -1.0  # 这里增加了两个参数
    args.best_epoch = 0
    acc_list = []
    # training
    for epoch in range(1, args.n_epoch):
        # train models
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list, train_loss = model.train(train_loader, epoch)

        # evaluate models
        test_acc1, test_acc2, test_acc, test_loss = model.evaluate(test_loader)

        mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)
        mean_pure_ratio = (mean_pure_ratio1 + mean_pure_ratio2) * 0.5

        if test_acc > args.best_acc:
            args.best_acc = test_acc
            args.best_epoch = epoch

            torch.save({'model_state_dict': model.model1.state_dict()}, "1epoch{}_acc{:.4f}.pth".format(epoch, test_acc))
            torch.save({'model_state_dict': model.model2.state_dict()}, "2epoch{}_acc{:.4f}.pth".format(epoch, test_acc))
            print("Saved Model!")

        logger.log({'epoch': epoch, 'train_acc1': train_acc1, 'train_acc2': train_acc2, 'train_loss': train_loss,
                    'test_acc1': test_acc1, 'test_acc2': test_acc2, 'test_acc': test_acc, 'test_loss': test_loss,
                    'mean pure ratio1': mean_pure_ratio1, 'mean pure ratio2': mean_pure_ratio2,
                    'best_test_acc': args.best_acc})

        with open(f'./results/ablation_study/{args.noise_type}_{args.noise_rate}_{args.name}' + '.txt', 'a') as f:
            f.write(str(epoch) + '_' + str(format(test_acc, '.2f')) + '\n')  # 保留2位小数

        with open(f'./results/ablation_study/{args.noise_type}_{args.noise_rate}_{args.name}_pure_ratio' + '.txt', 'a') as f:
            f.write(str(epoch) + '_' + str(format(mean_pure_ratio, '.2f')) + '\n')  # 保留2位小数

        if epoch >= 190:
            acc_list.extend([test_acc])

    avg_acc = sum(acc_list)/len(acc_list)
    print(len(acc_list))
    print(args.best_epoch, ": ", args.best_acc)
    print("the average acc in last 10 epochs: {}".format(str(avg_acc)))


if __name__ == '__main__':
    main()
