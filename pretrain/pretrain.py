import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import datetime
import numpy as np
import argparse
from data_provider_pretrain import DataProvider
from net_pretrain import MetaLearner

CLASSES_TRAIN = 64
CLASSES_VAL = 16
CLASSES_TEST = 20

def print_pretrain_info(args):
    print('>> GPU={}'.format(args.gpu_index))
    print('>> img size={}\tbatch size={}\tlr={}'.format(args.img_size, args.batch_size, args.lr_base))
    print('>> num epoch={}\tResNet=WRN-22-10'.format(args.num_epoch))
    print('>> save=\'../meta/weights/WRN_K{}_ceb{}.data\''.format(args.WRN_K, args.num_blocks))
    print('------------------------------------------------------------------------')

def main(args):
    if torch.cuda.is_available() is False:
        print('NO CUDA')
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index

    if os.path.exists('../meta/weights') is False:
        os.mkdir('../meta/weights')

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True # improve speed

    print_pretrain_info(args)

    model = MetaLearner(num_blocks=args.num_blocks, way=CLASSES_TRAIN+CLASSES_VAL, use_dropout=args.use_dropout,
                        use_leaky=args.use_leaky).to(device)
    data_provider_train = DataProvider(root_path=args.mini_imagenet_path, dataset_type='train', img_size=args.img_size,
                                       data_aug=True, mode='train')
    data_provider_test = DataProvider(root_path=args.mini_imagenet_path, dataset_type='test', img_size=args.img_size,
                                      data_aug=False, mode='test')

    data_loader_train = DataLoader(dataset=data_provider_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    data_loader_test = DataLoader(dataset=data_provider_test, batch_size=args.batch_size, shuffle=True, num_workers=8)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_base, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

    for epoch in range(1, args.num_epoch + 1):
        for i, (images, labels) in enumerate(data_loader_train):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            _, predict = torch.max(outputs.data, 1)
            total, correct = 0, 0
            total += labels.size(0)
            correct += (predict == labels).sum().item()
            acc = correct / total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = optimizer.state_dict()['param_groups'][0]['lr']
            print('\r>> GPU={}, Epoch=[{}/{}], Step=[{}/{}], Train Acc={:.4%}, Train Loss={:.4f}, Lr={:.4f}'.format(
                args.gpu_index, epoch, args.num_epoch, i+1, len(data_loader_train), acc, loss.item(), lr_), end='')

        print('')
        total, correct = 0, 0
        model.eval()
        with torch.no_grad():
            for i_, (images_, labels_) in enumerate(data_loader_test):
                print('\r>> Testing=[{}/{}]'.format(i_+1, len(data_loader_test)), end='')
                images_ = images_.to(device)
                labels_ = labels_.to(device)
                outputs = model(images_, bn_training=False)
                loss = F.cross_entropy(outputs, labels_)
                _, predict = torch.max(outputs.data, 1)
                total += labels_.size(0)
                correct += (predict == labels_).sum().item()
        print('>> GPU={}, Epoch=[{}/{}], Test Acc={:.4%}'.format(args.gpu_index, epoch, args.num_epoch, correct/total))
        model.train()

        if epoch % 10 == 0:
            torch.save(dict(model.state_dict()), '../meta/weights/WRN_K{}_ceb{}.data'.format(args.WRN_K, args.num_blocks))

        lr_scheduler.step()

    model.eval()
    print('')
    with torch.no_grad():
        correct, total = 0, 0
        for index, (images, labels) in enumerate(data_loader_test):
            if (index+1) % 10:
                print('\r>> Testing index=[{}/{}]'.format(index+1, len(data_loader_test)), end='')
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images, bn_training=False)
                _, predict = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        print('>> Test Accuracy={:.4%}'.format(correct/total))

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--mini_imagenet_path', type=str, help='dataset path', default='../data/miniimagenet')
    argparse.add_argument('--lr_base', type=float, help='base learning rate', default=0.1)
    argparse.add_argument('--batch_size', type=int, help='batch size', default=128)
    argparse.add_argument('--num_epoch', type=int, help='num of epoch', default=100)
    argparse.add_argument('--img_size', type=int, help='=images pixel', default=80)
    argparse.add_argument('--WRN_K', type=int, help='WRN depth scale', default=10)
    argparse.add_argument('--use_dropout', type=bool, help='wide-dropout', default=True)
    argparse.add_argument('--use_leaky', type=bool, help='leaky or ReLU', default=False)
    argparse.add_argument('--num_blocks', type=int, help='num of blocks', default=3)
    _args = argparse.parse_args()
    main(_args)















