import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


def distillation(student_scores, teacher_scores, T):
    p = F.log_softmax(student_scores / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)

    l_kl = F.kl_div(p, q, reduction="sum") * (T ** 2) / student_scores.shape[0]  # 求一个batch中所有样本的kl散度的总和，再求平均
    return l_kl


def loss_coco(y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda=0.1):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.data.cpu())
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]) / float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]) / float(num_remember)

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    loss1 = torch.mean(loss_1[ind_2_update])
    loss2 = torch.mean(loss_2[ind_1_update])

    loss = loss1 + loss2 + co_lambda * (distillation(y_1, y_2, T=1) + distillation(y_2, y_1, T=1))
    return loss, loss, pure_ratio_1, pure_ratio_2



def loss_coco_no_pure_ratio(y_1, y_2, t, forget_rate, co_lambda=0.1):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.data.cpu())
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    loss1 = torch.mean(loss_1[ind_2_update])
    loss2 = torch.mean(loss_2[ind_1_update])

    loss = loss1 + loss2 + co_lambda * (distillation(y_1, y_2, T=1) + distillation(y_2, y_1, T=1))
    return loss, loss