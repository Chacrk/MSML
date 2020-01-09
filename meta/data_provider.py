import os
import torch
import numpy as np
import random
import csv
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

class DataProvider(Dataset):
    def __init__(self, args, data_aug, total_batch_size, dataset_type):
        self.path = args.mini_imagenet_path
        self.data_aug = data_aug
        self.total_batch_size = total_batch_size
        self.way = args.way
        self.k_shot = args.k_shot
        self.k_query = args.k_query
        self.img_size = args.img_size
        self.dataset_type = dataset_type

        if os.path.exists('{}/images'.format(self.root_path)) is False:
            raise Exception('dataset not found')
        self.num_inputa = self.way * self.k_shot
        self.num_inputb = self.way * self.k_query
        self.images_path = '{}/images'.format(args.mini_imagenet_path)
        self.filenames_by_label = self.make_filenames_by_label('{}/{}.csv'.format(self.path, self.dataset_type))
        self.filenames_by_index = []
        self.index_by_label = {}

        for i, (key_, values_) in enumerate(self.filenames_by_label.items()):
            self.filenames_by_index.append(values_)
            self.index_by_label[key_] = i
        self.total_num_classes = len(self.filenames_by_index)
        self.create_batches(self.total_batch_size)

    def make_filenames_by_label(self, csv_path):
        filenames_by_label = {}
        with open(csv_path) as csv_:
            csv_reader = csv.reader(csv_)
            next(csv_reader)
            for i, row_item in enumerate(csv_reader):
                if row_item[1] not in filenames_by_label.keys():
                    filenames_by_label[row_item[1]] = []
                filenames_by_label[row_item[1]].append(row_item[0])

        if self.dataset_type == 'train':
            with open('{}/{}.csv'.format(self.path, 'val')) as csv_:
                csv_reader = csv.reader(csv_)
                next(csv_reader)
                for i, row_item in enumerate(csv_reader):
                    if row_item[1] not in filenames_by_label.keys():
                        filenames_by_label[row_item[1]] = []
                    filenames_by_label[row_item[1]].append(row_item[0])
        return filenames_by_label

    def create_batches(self, total_batch_size):
        self.inputa_batches = []
        self.inputb_batches = []

        for batch_index in range(total_batch_size):
            if (batch_index+1) % 10 == 0:
                print('\r>> Generating {} tasks=[{}/{}]'.format(self.dataset_type, batch_index+1,
                                                                total_batch_size), end='')
            selected_classes = np.random.choice(self.total_num_classes, self.way, replace=False)
            np.random.shuffle(selected_classes)

            inputa_tmp, inputb_tmp = [], []
            for class_index in selected_classes:
                selected_img_index = np.random.choice(len(self.filenames_by_index[class_index]),
                                                      self.k_shot + self.k_query, replace=False)
                np.random.shuffle(selected_img_index)
                index_in_inputa = np.array(selected_img_index[: self.k_shot])
                index_in_inputb = np.array(selected_img_index[self.k_shot: ])

                inputa_tmp.append(np.array(self.filenames_by_index[class_index])[index_in_inputa].tolist())
                inputb_tmp.append(np.array(self.filenames_by_index[class_index])[index_in_inputb].tolist())

            random.shuffle(inputa_tmp)
            random.shuffle(inputb_tmp)
            self.inputa_batches.append(inputa_tmp)
            self.inputb_batches.append(inputb_tmp)
        print('')

    def __getitem__(self, index):
        inputa = torch.FloatTensor(self.way * self.k_shot, 3, self.img_size, self.img_size)
        inputb = torch.FloatTensor(self.way * self.k_query, 3, self.img_size, self.img_size)

        inputa_flatten = ['{}/{}'.format(self.images_path, item) for sub in self.inputa_batches[index] for item in sub]
        inputb_flatten = ['{}/{}'.format(self.images_path, item) for sub in self.inputb_batches[index] for item in sub]
        labela = [self.index_by_label[item[:9]] for sub in self.inputa_batches[index] for item in sub]
        labelb = [self.index_by_label[item[:9]] for sub in self.inputb_batches[index] for item in sub]

        label_unique = np.unique(labela)
        random.shuffle(label_unique)

        labela_relative = np.zeros(self.num_inputa)
        labelb_relative = np.zeros(self.num_inputb)

        for index_, lu in enumerate(label_unique):
            labela_relative[labela == lu] = index_
            labelb_relative[labelb == lu] = index_

        self.mean = np.array([x / 255.0 for x in [125.3, 123.0, 113.9]])
        self.std = np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])

        if self.data_aug:
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize(92),
                # transforms.RandomResizedCrop(80, scale=(0.8, 0.9)),
                transforms.RandomResizedCrop(80),
                # transforms.RandomResizedCrop(88),
                # transforms.CenterCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])
        else:
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize(92),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])

        for i, path in enumerate(inputa_flatten):
            inputa[i] = self.transform(path)

        for i, path in enumerate(inputb_flatten):
            inputb[i] = self.transform(path)

    def __len__(self):
        return self.total_batch_size


























