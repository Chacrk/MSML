import torch
import os
import numpy as np
from torch.nn import functional as F
from copy import deepcopy
import torch.nn as nn
from net import MetaLearner

class Meta(torch.nn.Module):
    def __init__(self, args):
        super(Meta, self).__init__()
        self.way = args.way
        self.k_show = args.k_shot
        self.k_query = args.k_query
        self.batch_size = self.batch_size
        self.num_inner_updates = args.num_inner_updates
        self.num_inner_updates_test = args.num_inner_updates_test
        self.num_blocks = args.num_blocks
        self.WRN_K = args.WRN_K

        self.lr_decay = 0.99
        self.meta_learner = MetaLearner(num_blocks=self.num_blocks, way=self.way, use_dropout=args.use_dropout,
                                        use_leaky=args.use_leaky)
        self.all_params_name_list = self.meta_learner.dict_parameters().keys()
        self.keys_inner = list(filter(lambda x: 'fc' in x, self.all_params_name_list))
        self.meta_learner.define_task_lr(self.keys_inner)

        pretrained_weight = 'weights/WRN_K{}_ceb{}.data'.format(self.WRN_K, self.num_blocks)
        if os.path.exists(pretrained_weight) is False:
            raise Exception('Pretrain weight not exists')

        pretrained_dict_data = torch.load(pretrained_weight)
        self.model_dict = self.meta_learner.state_dict()
        processed_pretrained_dict = {key: value for key, value in pretrained_dict_data.items()
                                     if (key in self.model_dict and 'fc' not in key)}
        self.model_dict.update(processed_pretrained_dict)
        self.meta_learner.load_state_dict(self.model_dict)
        self.keys_outer = list(filter(lambda x: 'bias' in x or 'norm' in x or '2/filter' in x, self.all_params_name_list))
        self.vars_conv_norm = list(filter(lambda x: 'fc' not in x, self.keys_outer))

        self.optim = torch.optim.Adam(
            [{'params': self.meta_learner.parameters_by_names(self.vars_conv_norm), 'lr': 1e-4}], lr=1e-3)
        self.le_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1000, gamma=0.6)
        self.lr_fc = 1e-2
        self.itrs = 0

    def load_last_weight(self):
        last_dict_file = 'weights/WRN_K{}_ceb{}_last.data'.format(self.WRN_K, self.num_blocks)
        dict_data = torch.load(last_dict_file)
        self.model_dict = self.meta_learner.state_dict()
        pretrained_dict = {key: value for key, value in dict_data.items() if key in self.model_dict}
        self.model_dict.update(pretrained_dict)
        self.meta_learner.load_state_dict(self.model_dict)

    def save_model(self):
        torch.save(dict(self.meta_learner.state_dict()), 'weights/WRN_K{}_ceb{}_last.data'.format(self.WRN_K, self.num_blocks))

    def forward(self, inputa, labela, inputb, labelb):
        dropout_training = True

        loss_last = 0.0
        correct_last = 0

        ckps = [20, 50]
        ckp_rates = {'20': 0.1, '50': 0.2}
        ckp_losses = 0.0

        for i in range(self.batch_size):
            output_a = self.meta_learner(inputa[i], weights=None, bn_training=True, dropout_training=dropout_training)
            loss_a = F.cross_entropy(output_a, labela[i])
            grads = torch.autograd.grad(loss_a, self.meta_learner.parameters_by_names(self.keys_inner))
            grad_fc_w = grads[0] - torch.matmul(torch.matmul(self.meta_learner.dict_parameters()['fc/weight'],
                                                             grads[0].t()),
                                                self.meta_learner.dict_parameters()['fc/weight'])
            fast_weights_value = list(map(lambda p: p[1][1] - self.meta_learner.task_lr[p[1][0]] * p[0],
                                          zip([grad_fc_w, grads[1]],
                                              self.meta_learner.parameters_inner_item(self.keys_inner))))
            fast_weights = dict(zip(self.keys_inner, fast_weights_value))

            for k in range(1, self.num_inner_updates):
                output_a = self.meta_learner(inputa[i], weights=fast_weights, bn_training=True,
                                             dropout_training=dropout_training)
                loss_a = F.cross_entropy(output_a, labela[i])
                grads = torch.autograd.grad(loss_a, list(fast_weights.values()))
                grad_fc_w = grads[0] - torch.matmul(torch.matmul(fast_weights['fc/weight'], grads[0].t()),
                                                    fast_weights['fc/weight'])
                fast_weights_value = list(
                    map(lambda p: p[1][1] - (self.lr_decay ** k) * self.meta_learner.task_lr[p[1][0]] * p[0],
                        zip([grad_fc_w, grads[1]], fast_weights.items())))
                fast_weights = dict(zip(self.keys_inner, fast_weights_value))

                if k in ckps:
                    output_b = self.meta_learner(inputb[i], weights=fast_weights, bn_training=True,
                                                 dropout_training=dropout_training)
                    ckp_losses += ckp_rates[str(k)] * F.cross_entropy(output_b, labelb[i])

                if k == self.num_inner_updates - 1:
                    output_b = self.meta_learner(inputb[i], weights=fast_weights, bn_training=True,
                                                 dropout_training=dropout_training)
                    loss_last = F.cross_entropy(output_b, labelb[i])
                    with torch.no_grad():
                        pred_b = F.softmax(output_b, dim=1).argmax(dim=1)
                        correct_last = torch.eq(pred_b, labelb[i]).sum().item()

        loss_final = ((1.0 - np.sum(list(ckp_rates.values()))) * loss_last + ckp_losses) / self.batch_size
        grad_loss_final_fc = torch.autograd.grad(loss_final,
                                                 self.meta_learner.parameters_by_names(self.keys_inner),
                                                 retain_graph=True)
        grad_fc_w = grad_loss_final_fc[0] - torch.matmul(
            torch.matmul(self.meta_learner.dict_parameters()['fc/weight'], grad_loss_final_fc[0].t()),
            self.meta_learner.dict_parameters()['fc/weight'])
        self.itrs += 1
        if (self.itrs + 1) % 1000 == 0:
            self.lr_fc *= 0.6
        fast_weights_value = list(map(lambda p: p[1][1] - self.lr_fc * p[0],
                                      zip([grad_fc_w, grad_loss_final_fc[1]],
                                          self.meta_learner.parameters_inner_item(self.keys_inner))))
        self.meta_learner.dict_parameters()['fc/weight'].data = fast_weights_value[0]
        self.meta_learner.dict_parameters()['fc/bias'].data = fast_weights_value[1]

        self.optim.zero_grad()
        loss_final.backward()
        self.optim.step()
        self.lr_scheduler.step()

        acc_last_r = correct_last / (labelb.size(1) * self.batch_size)
        loss_last_r = loss_last.item()
        return acc_last_r, loss_last_r

    def forward_test(self, inputa, labela, inputb, labelb):
        loss_last = 0.0
        correct_last = 0

        output_a = self.meta_learner(inputa, weights=None, bn_training=True, droput_training=False)
        loss_a = F.cross_entropy(output_a, labela)
        grads = torch.autograd.grad(loss_a, self.meta_learner.parameters_by_names(self.keys_inner))
        grad_fc_w = grads[0] - torch.matmul(torch.matmul(self.meta_learner.dict_parameters()['fc/weight'],
                                                         grads[0].t()),
                                            self.meta_learner.dict_parameters()['fc/weight'])
        fast_weights_value = list(map(lambda p: p[1][1] - self.meta_learner.task_lr[p[1][0]] * p[0],
                                      zip([grad_fc_w, grads[1]],
                                          self.meta_learner.parameters_inner_item(self.keys_inner))))
        fast_weights = dict(zip(self.keys_inner, fast_weights_value))

        for k in range(1, self.num_inner_updates_test):
            output_a = self.meta_learner(inputa, weights=fast_weights, bn_training=True, droput_training=False)
            loss_a = F.cross_entropy(output_a, labela)
            grads = torch.autograd.grad(loss_a, list(fast_weights.values()))
            grad_fc_w = grads[0] - torch.matmul(torch.matmul(fast_weights['fc/weight'], grads[0].t()),
                                                fast_weights['fc/weight'])
            grads_st = [grad_fc_w, grads[1]]
            fast_weights_value = list(
                map(lambda p: p[1][1] - (self.lr_decay ** k) * self.meta_learner.task_lr[p[1][0]] * p[0],
                    zip(grads_st, fast_weights.items())))
            fast_weights = dict(zip(self.keys_inner, fast_weights_value))

            if k == self.num_inner_updates_test - 1:
                with torch.no_grad():
                    output_b = self.meta_learner(inputb, weights=fast_weights, bn_training=True, droput_training=False)
                    loss_last = F.cross_entropy(output_b, labelb).item()
                    pred_b = F.softmax(output_b, dim=1).argmax(dim=1)
                    correct_last = torch.eq(pred_b, labelb).sum().item()
        acc_r = correct_last / inputb.size(0)

        return acc_r, loss_last























