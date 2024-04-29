# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.cnn import MLPNet, CNN
import numpy as np
from common.utils import accuracy, sigmoid_rampup

from algorithm.loss import loss_coco


class CoCo:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = 128
        learning_rate = args.lr

        if args.forget_rate is None:
            if args.noise_type == "asymmetric":
                forget_rate = args.noise_rate / 2
            else:
                forget_rate = args.noise_rate
        else:
            forget_rate = args.forget_rate

        self.noise_or_not = train_dataset.noise_or_not

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.9
        self.alpha_plan = [learning_rate] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch

        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        self.co_lambda = args.co_lambda
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset

        if args.model_type == "cnn":
            self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
            self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        elif args.model_type == "mlp":
            self.model1 = MLPNet()
            self.model2 = MLPNet()

        self.model1.to(device)
        self.model2.to(device)
        self.model1 = torch.nn.DataParallel(self.model1)
        self.model2 = torch.nn.DataParallel(self.model2)

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                          lr=learning_rate)

        self.loss_fn = loss_coco

        self.rampup_length = args.rampup_length
        self.phase_shift = args.phase_shift
        self.alpha = args.alpha

    # Evaluate the Model
    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        correct1 = 0
        correct2 = 0
        correct = 0
        total = 0
        running_loss = 0.0
        for images, _, labels, _ in test_loader:
            images = Variable(images).to(self.device)

            logits1 = self.model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            correct1 += (pred1.cpu() == labels).sum()

            logits2 = self.model2(images)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)
            correct2 += (pred2.cpu() == labels).sum()

            logits = logits1 + logits2
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            correct += (pred.cpu() == labels).sum()

            total += labels.size(0)

            loss = torch.nn.CrossEntropyLoss()(logits.cpu(), labels)
            running_loss += loss.item()

        acc1 = 100 * float(correct1) / float(total)
        acc2 = 100 * float(correct2) / float(total)
        acc = 100 * float(correct) / float(total)
        running_loss /= len(test_loader)
        print("Test Acc: {} %".format(acc))
        return acc1, acc2, acc, running_loss

    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode

        self.adjust_learning_rate(self.optimizer, epoch)

        running_loss = 0.0
        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        for i, (images, images1, labels, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
            if i > self.num_iter_per_epoch:
                break

            images = Variable(images).to(self.device)
            images1 = Variable(images1).to(self.device)
            labels = Variable(labels).to(self.device)

            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2(images1)
            prec2 = accuracy(logits2, labels, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2

            alpha = sigmoid_rampup(epoch, self.rampup_length, self.phase_shift) * self.alpha

            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch],
                                                                      ind, self.noise_or_not, alpha)

            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            running_loss += loss_1.item()

            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)

            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
                       loss_1.data.item(), loss_2.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list), sum(pure_ratio_2_list) / len(pure_ratio_2_list)))

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)
        running_loss /= len(train_loader)
        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list, running_loss

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
