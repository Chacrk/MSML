import torch
import numpy as np
import os
import csv
from torch.utils.data import Dataset
from torchvision.transforms import transforms, Normalize
from PIL import Image
import random

class DataProvider(Dataset):
    def __init__(self, root_path, dataset_type, img_size, data_aug, mode):
        self.root_path = root_path
        self.dataset_type = dataset_type
        self.img_size = img_size
        self.mode = mode
        self.data_aug = data_aug

        self.image_name_list_base = []
        self.index_list_base = []
        self.image_name_list = []
        self.index_list = []

        if os.path.exists('{}/images'.format(self.root_path)) is False:
            raise Exception('dataset not found')

        self.images_path = '{}/images'.format(self.root_path)

        if self.dataset_type == 'train':
            self.num_classes = 16 + 64
            self.build_file_list('{}/{}'.format(root_path, 'train'), '{}/{}.csv'.format(root_path, 'val'))
        else:
            self.num_classes = 20
            self.build_file_list('{}/{}.csv'.format(root_path, 'test'), '')

    def build_file_list(self, csv_path_1, csv_path_2):
        if self.dataset_type == 'train':
            with open(csv_path_1) as csv_1:
                csv_reader = csv.reader(csv_1)
                next(csv_reader)
                for index, row_item in enumerate(csv_reader):
                    self.image_name_list_base.append(row_item[0])
                    self.index_list_base.append(index - 1)
            last_index = self.index_list_base[-1]
            with open(csv_path_2) as csv_2:
                csv_reader = csv.reader(csv_2)
                next(csv_reader)
                for index, row_item in enumerate(csv_reader):
                    self.image_name_list_base.append(row_item[0])
                    self.index_list_base.append(last_index + index - 1)
        else:
            with open(csv_path_1) as csv_1:
                csv_reader = csv.reader(csv_1)
                next(csv_reader)
                for index, row_item in enumerate(csv_reader):
                    self.image_name_list_base.append(row_item[0])
                    self.index_list_base.append(index - 1)

        self.label_list_base = [i for i in range(self.num_classes) for _ in range(600)]
        random.shuffle(self.index_list_base)
        self.image_name_list = [self.image_name_list_base[i] for i in self.index_list_base]
        self.label_list = [self.index_list_base[i] for i in self.index_list_base]

        self.mean = np.array([x / 255.0 for x in [125.3, 123.0, 113.9]])
        self.std = np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])

        if self.data_aug is True:
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.RandomResizedCrop(88),
                transforms.CenterCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, item):
        filename = self.image_name_list[item]
        label = self.label_list[item]
        image = self.transform(Image.open('{}/{}'.format(self.images_path, filename)).convert('RGB'))
        return image, label



















