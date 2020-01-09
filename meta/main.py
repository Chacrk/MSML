import torch
import os
import datetime
import numpy as np
from  torch.utils.data import DataLoader
import argparse
from model import Meta
from data_provider import DataProvider

def print_meta_info(args):
    print('>> GPU={}'.format(args.gpu_index))
    print('>> k shot={}\tk query={}\timg size={}\tbatch size={}\tepoch index={}'.format(
        args.k_shot, args.k_query, args.img_size, args.batch_size, args.epoch_index))
    print('>> num inner loop={}\tnum inner loop test={}\tResNet=WRN-22-10'.format(
        args.num_inner_updates, args.num_inner_updates_test))
    print('>> pretrain data=\'weights/WRN_K{}_ceb{}.data\''.format(args.WRN_K, args.num_blocks))
    print('>> save model=\'weights/WRN_K{}_ceb{}_last.data\''.format(args.WRN_K, args.num_blocks))
    print('------------------------------------------------------------------------')


def main(args):
    if torch.cuda.is_available() is False:
        print('NO CUDA')
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True # improve speed

    print_meta_info(args)

    model = Meta(args=args).to(device)

    data_provider_train = DataProvider(args=args, data_aug=True, total_batch_size=15501, dataset_type='train')
    data_provider_val = DataProvider(args=args, data_aug=False, total_batch_size=400, dataset_type='test')
    data_provider_test = DataProvider(args=args, data_aug=False, total_batch_size=600, dataset_type='test')

    data_loader_train = DataLoader(data_provider_train, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                   pin_memory=True)
    for step, (inputa, labela, inputb, labelb) in enumerate(data_loader_train):
        print('\rTraining step: [{}/{}]'.format(step, len(data_loader_train)), end='')
        inputa = inputa.to(device)
        labela = labela.to(device)
        inputb = inputb.to(device)
        labelb = labelb.to(device)

        train_acc, train_loss = model(inputa, labela, inputb, labelb)

        if (step+1) % 100 == 0:
            print('\rTrain Acc={:.3%}, Train Loss={:.3f}'.format(train_acc, train_loss), end='')
            print('--------------------------------------------')

        if (step+1) % 500 == 0:
            print('')
            data_loader_val = DataLoader(data_provider_val, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
            val_acc_list = []
            val_loss_list = []
            for index, (inputa_, labela_, inputb_, labelb_) in enumerate(data_loader_val):
                print('\rVal step=[{}/{}]'.format(index+1, len(data_loader_val)), end='')
                inputa_ = inputa_.squeeze_().to(device)
                labela_ = labela_.squeeze_().to(device)
                inputb_ = inputb_.squeeze_().to(device)
                labelb_ = labelb_.squeeze_().to(device)
                val_acc, val_loss = model.forward_test(inputa_, labela_, inputb_, labelb_)
                val_acc_list.append(val_acc)
                val_loss_list.append(val_loss)

            val_acc_mean = np.array(val_acc_list).mean()
            val_loss_mean = np.array(val_loss_list).mean()
            print('>> Val Acc={:.4%}, Val Loss={:.4f}'.format(val_acc_mean, val_loss_mean))
            model.save_model()

        model.save_model()

    data_loader_test = DataLoader(data_provider_test, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    model.load_last_weight()
    test_acc_list, test_loss_list = [], []
    for index, (inputa, labela, inputb, labelb) in enumerate(data_loader_test):
        print('\rTesting: [{}/{}]'.format(index + 1, len(data_loader_test)), end='')
        inputa = inputa.squeeze_().to(device)
        labela = labela.squeeze_().to(device)
        inputb = inputb.squeeze_().to(device)
        labelb = labelb.squeeze_().to(device)
        test_acc, test_loss = model.forward_test(inputa, labela, inputb, labelb)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
    print('')

    test_acces_list = np.array(test_acc_list).mean()
    test_losses_list = np.array(test_loss_list).mean()
    print('>> Acc={:.4%}, Loss={:.4f}'.format(test_acces_list, test_losses_list))

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    # network structure
    argparse.add_argument('--WRN_K', type=int, help='WRN depth scale', default=10)
    argparse.add_argument('--use_dropout', type=bool, help='use dropout or not', default=True)
    argparse.add_argument('--use_leaky', type=bool, help='acti func', default=False)
    argparse.add_argument('--num_blocks', type=int, help='num of blocks', default=3)
    # meta-train
    argparse.add_argument('--itrs', type=int, help='轮数', default=60000)
    argparse.add_argument('--mini_imagenet_path', type=str, help='dataset path', default='../data/miniimagenet')
    argparse.add_argument('--way', type=int, help='way', default=5)
    argparse.add_argument('--k_shot', type=int, help='K', default=1)
    argparse.add_argument('--k_query', type=int, help='K query', default=15)
    argparse.add_argument('--img_size', type=int, help='image_size', default=80)
    argparse.add_argument('--img_channels', type=int, help='image channel', default=3)
    argparse.add_argument('--batch_size', type=int, help='num of tasks in batch', default=1)
    argparse.add_argument('--inner_lr', type=float, help='lr in inner loop', default=0.01)
    argparse.add_argument('--outer_lr', type=float, help='lr in outer loop', default=0.001)
    argparse.add_argument('--num_inner_updates', type=int, help='update num during meta-train', default=100)
    argparse.add_argument('--num_inner_updates_test', type=int, help='update num during meta-test', default=100)
    argparse.add_argument('--epoch_index', type=int, help='', default=80)

    # data
    argparse.add_argument('--use_sti', type=bool, help='使用Stiefel或欧几里得', default=True)
    argparse.add_argument('--use_stage', type=bool, help='是否使用多阶段联合优化', default=True)

    _args = argparse.parse_args()
    main(_args)





















