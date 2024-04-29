from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from torchvision import transforms


# 已经做了数据增广
from common.utils import KCropsTransform, MixTransform
from common.autoaugment import RandAugment


class animal_dataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        train_path = os.listdir(os.path.abspath(root) + '/training')
        test_path = os.listdir(os.path.abspath(root) + '/testing')
        # print(train_path)
        # print('Please be patient for image loading!')
        if mode == 'train':
            dir_path = os.path.abspath(root) + '/training'
            self.targets = [int(i.split('_')[0]) for i in train_path]
            self.data = [np.asarray(Image.open(dir_path + '/' + i)) for i in train_path]
        else:
            dir_path = os.path.abspath(root) + '/testing'
            self.targets = [int(i.split('_')[0]) for i in test_path]
            self.data = [np.asarray(Image.open(dir_path + '/' + i)) for i in test_path]
        # print('Loading finished!')

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # img1 = transforms.RandomHorizontalFlip(p=1)(img)

        return img, target  # , index

    def update_labels(self, new_label):
        self.targets = new_label.cpu()

    def __len__(self):
        return len(self.targets)


class animal_dataloader():
    def __init__(self, batch_size, num_workers):

        self.batch_size = batch_size
        # self.num_class = num_class
        self.num_workers = num_workers

    def run(self):
        strong_transform = transforms.Compose([transforms.RandomCrop(64, padding=8),
                                               transforms.RandomHorizontalFlip(),
                                               RandAugment(),
                                               transforms.ToTensor()])
        weak_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        none_transform = transforms.Compose([transforms.ToTensor()])

        root_dir = '/home/pris/wwj/datasets/animal-10n/'

        train_dataset = animal_dataset(root=root_dir, transform=KCropsTransform(strong_transform, 2), mode='train')
        test_dataset = animal_dataset(root=root_dir, transform=none_transform,  mode='test')

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)

        return train_loader, test_loader