import torch
from torch.nn import functional as F
from torch import nn
import torch.utils.data
from torch.nn.functional import pad
from torch.nn.modules.utils import _pair

class MetaLearner(nn.Module):
    def __init__(self, num_blocks, way, use_dropout, use_leaky):
        super(MetaLearner, self).__init__()
        self.weights = nn.ParameterDict()
        self.weights_br = nn.ParameterDict() # running mean & running var
        self.use_conv_bias = True

        # hyper parameters
        self.channels_queue_base = [8, 16, 32, 64]
        self.WRN_K = 10
        self.num_blocks = num_blocks
        self.channels_queue = [x * self.WRN_K for x in self.channels_queue_base]
        if use_leaky:
            self.relu = F.leaky_relu
        else:
            self.relu = F.relu
        self.dropout_p = 0.5
        self.use_dropout = use_dropout

        # meta-SGD
        self.task_lr = nn.ParameterDict()

        # first part # [out, in, ker, ker]
        self.weights['first_conv0/filter'] = nn.Parameter(torch.ones([self.channels_queue[0], 3, 3, 3]), False)
        self.weights['first_conv0/bias']   = nn.Parameter(torch.zeros(self.channels_queue[0]), True)
        self.weights['first_norm0/weight'] = nn.Parameter(torch.ones(self.channels_queue[0]), True)
        self.weights['first_norm0/bias'] = nn.Parameter(torch.zeros(self.channels_queue[0]), True)
        self.weights_br['first_norm0/mean'] = nn.Parameter(torch.zeros(self.channels_queue[0]), False)
        self.weights_br['first_norm0/var'] = nn.Parameter(torch.zeros(self.channels_queue[0]), False)

        # blocks part
        for i in range(len(self.channels_queue) - 1):
            self.construct_block_weights(i + 1, in_channel=self.channels_queue[i], out_channel=self.channels_queue[i+1])

        self.weights['fc/weight'] = nn.Parameter(torch.ones([way, 640]), requires_grad=True)
        self.weights['fc/bias'] = nn.Parameter(torch.zeros(way), requires_grad=True)

        # init classify layer params
        for s_ in self.named_parameters():
            if 'fc/weight' in s_[0]:
                nn.init.orthogonal_(s_[1])

    def zero_init_fc_weight(self):
        for s_ in self.named_parameters():
            if 'fc/weight' in s_[0]:
                nn.init.zeros_(s_[1])

    def kaiming_init_fc_weight(self):
        for s_ in self.named_parameters():
            if 'fc/weight' in s_[0]:
                nn.init.kaiming_uniform_(s_[1])

    def define_task_lr(self, vars_to_optim):
        self.task_lr['fc/weight'] = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)
        self.task_lr['fc/bias'] = nn.Parameter(torch.tensor(0.03, dtype=torch.float32), requires_grad=True)
        # for key, value in self.named_parameters():
        #     if key in vars_to_optim:
        #         self.task_lr[key] = nn.Parameter(torch.tensor(0.05, dtype=torch.float32), requires_grad=False)

    def construct_block_weights(self, block_index, in_channel, out_channel):
        # [out, in, ker_size, ker_size]
        self.weights['b{}_0/conv0/filter'.format(block_index)] = nn.Parameter(torch.ones([out_channel, in_channel, 3, 3]), False)
        self.weights['b{}_0/conv0/bias'.format(block_index)] = nn.Parameter(torch.zeros(out_channel), True)
        self.weights['b{}_0/norm0/weight'.format(block_index)] = nn.Parameter(torch.ones(out_channel), True)
        self.weights['b{}_0/norm0/bias'.format(block_index)] = nn.Parameter(torch.zeros(out_channel), True)
        self.weights_br['b{}_0/norm0/mean'.format(block_index)] = nn.Parameter(torch.zeros(out_channel), False)
        self.weights_br['b{}_0/norm0/var'.format(block_index)] = nn.Parameter(torch.zeros(out_channel), False)

        self.weights['b{}_0/conv1/filter'.format(block_index)] = nn.Parameter(torch.ones([out_channel, out_channel, 3, 3]), False)
        self.weights['b{}_0/conv1/bias'.format(block_index)] = nn.Parameter(torch.zeros(out_channel), True)
        self.weights['b{}_0/norm1/weight'.format(block_index)] = nn.Parameter(torch.ones(out_channel), True)
        self.weights['b{}_0/norm1/bias'.format(block_index)] = nn.Parameter(torch.zeros(out_channel), True)
        self.weights_br['b{}_0/norm1/mean'.format(block_index)] = nn.Parameter(torch.zeros(out_channel), False)
        self.weights_br['b{}_0/norm1/var'.format(block_index)] = nn.Parameter(torch.zeros(out_channel), False)

        # shortcut
        self.weights['b{}_0/conv2/filter'.format(block_index)] = nn.Parameter(torch.ones([out_channel, in_channel, 1, 1]), True)
        self.weights['b{}_0/conv2/bias'.format(block_index)] = nn.Parameter(torch.zeros(out_channel), True)  # shortcut's bias
        self.weights['b{}_0/norm2/weight'.format(block_index)] = nn.Parameter(torch.ones(out_channel), True)
        self.weights['b{}_0/norm2/bias'.format(block_index)] = nn.Parameter(torch.zeros(out_channel), True)
        self.weights_br['b{}_0/norm2/mean'.format(block_index)] = nn.Parameter(torch.zeros(out_channel), False)
        self.weights_br['b{}_0/norm2/var'.format(block_index)] = nn.Parameter(torch.zeros(out_channel), False)

        for i in range(1, self.num_blocks):
            self.weights['b{}_{}/conv0/filter'.format(block_index, i)] = nn.Parameter(torch.ones([out_channel, out_channel, 3, 3]), False)
            self.weights['b{}_{}/conv0/bias'.format(block_index, i)] = nn.Parameter(torch.zeros(out_channel), True)
            self.weights['b{}_{}/norm0/weight'.format(block_index, i)] = nn.Parameter(torch.ones(out_channel), True)
            self.weights['b{}_{}/norm0/bias'.format(block_index, i)] = nn.Parameter(torch.zeros(out_channel), True)
            self.weights_br['b{}_{}/norm0/mean'.format(block_index, i)] = nn.Parameter(torch.zeros(out_channel), False)
            self.weights_br['b{}_{}/norm0/var'.format(block_index, i)] = nn.Parameter(torch.zeros(out_channel), False)

            self.weights['b{}_{}/conv1/filter'.format(block_index, i)] = nn.Parameter(torch.ones([out_channel, out_channel, 3, 3]), False)
            self.weights['b{}_{}/conv1/bias'.format(block_index, i)] = nn.Parameter(torch.zeros(out_channel), True)
            self.weights['b{}_{}/norm1/weight'.format(block_index, i)] = nn.Parameter(torch.ones(out_channel), True)
            self.weights['b{}_{}/norm1/bias'.format(block_index, i)] = nn.Parameter(torch.zeros(out_channel), True)
            self.weights_br['b{}_{}/norm1/mean'.format(block_index, i)] = nn.Parameter(torch.zeros(out_channel), False)
            self.weights_br['b{}_{}/norm1/var'.format(block_index, i)] = nn.Parameter(torch.zeros(out_channel), False)

    def conv2d_tf_style(self, input_, weight, bias=None, stride=1, padding='SAME', dilation=1, groups=1):
        stride = _pair(stride)
        dilation = _pair(dilation)

        def check_format(*argv):
            argv_format = []

            for i in range(len(argv)):
                if type(argv[i]) is int:
                    argv_format.append((argv[i], argv[i]))
                elif hasattr(argv[i], "__getitem__"):
                    argv_format.append(tuple(argv[i]))
                else:
                    raise TypeError('all input should be int or list-type')
            return argv_format

        stride, dilation = check_format(stride, dilation)

        if padding == 'SAME':
            padding = 0
            input_rows = input_.size(2)
            filter_rows = weight.size(2)
            out_rows = (input_rows + stride[0] - 1) // stride[0]
            padding_rows = max(0, (out_rows - 1) * stride[0] +
                               (filter_rows - 1) * dilation[0] + 1 - input_rows)
            rows_odd = padding_rows % 2
            input_cols = input_.size(3)
            filter_cols = weight.size(3)
            out_cols = (input_cols + stride[1] - 1) // stride[1]
            padding_cols = max(0, (out_cols - 1) * stride[1] +
                               (filter_cols - 1) * dilation[1] + 1 - input_cols)
            cols_odd = padding_cols % 2
            input_ = pad(input_, [padding_cols // 2, padding_cols // 2 + int(cols_odd),
                                  padding_rows // 2, padding_rows // 2 + int(rows_odd)])
        elif padding == 'VALID':
            padding = 0

        elif type(padding) != int:
            raise ValueError('Padding should be \'SAME\', \'VALID\'.'.format(padding))

        return F.conv2d(input_, weight, bias, stride, padding=padding, dilation=dilation, groups=groups)

    def swish(self, x):
        return torch.sigmoid(x) * x

    def conv_norm_activation(self, input_, weights, bn_training):
        x = F.conv2d(input_, weight=weights['first_conv0/filter'], bias=weights['first_conv0/bias'], stride=1, padding=1)
        x = F.batch_norm(input=x, running_mean=self.weights_br['first_norm0/mean'],
                         running_var=self.weights_br['first_norm0/var'], weight=weights['first_norm0/weight'],
                         bias=weights['first_norm0/bias'], training=bn_training, momentum=0.1)
        x = self.relu(x, inplace=True)
        return x

    def expand_conv(self, input_, weights, block_index, bn_training, droput_training):
        if self.use_conv_bias:
            x = F.conv2d(input=input_, weight=weights['b{}_0/conv0/filter'.format(block_index)],
                         bias=weights['b{}_0/conv0/bias'.format(block_index)], stride=2, padding=1)
        else:
            x = F.conv2d(input=input_, weight=weights['b{}_0/conv0/filter'.format(block_index)],
                         bias=None, stride=2, padding=1)
        x = F.batch_norm(input=x, running_mean=self.weights_br['b{}_0/norm0/mean'.format(block_index)],
                         running_var=self.weights_br['b{}_0/norm0/var'.format(block_index)],
                         weight=weights['b{}_0/norm0/weight'.format(block_index)],
                         bias=weights['b{}_0/norm0/bias'.format(block_index)], training=bn_training, momentum=0.1)
        x = self.relu(x, inplace=True)

        if self.use_dropout: x = F.dropout(input=x, p=self.dropout_p, training=droput_training)

        if self.use_conv_bias:
            x = F.conv2d(input=x, weight=weights['b{}_0/conv1/filter'.format(block_index)],
                         bias=weights['b{}_0/conv1/bias'.format(block_index)], stride=1, padding=1)
        else:
            x = F.conv2d(input=x, weight=weights['b{}_0/conv1/filter'.format(block_index)],
                         bias=None, stride=1, padding=1)
        x = F.batch_norm(input=x, running_mean=self.weights_br['b{}_0/norm1/mean'.format(block_index)],
                         running_var=self.weights_br['b{}_0/norm1/var'.format(block_index)],
                         weight=weights['b{}_0/norm1/weight'.format(block_index)],
                         bias=weights['b{}_0/norm1/bias'.format(block_index)], training=bn_training, momentum=0.1)
        if self.use_conv_bias:
            shortcut = F.conv2d(input=input_, weight=weights['b{}_0/conv2/filter'.format(block_index)],
                                bias=weights['b{}_0/conv2/bias'.format(block_index)], stride=2)
        else:
            shortcut = F.conv2d(input=input_, weight=weights['b{}_0/conv2/filter'.format(block_index)],
                                bias=None, stride=2)
        shortcut = F.batch_norm(input=shortcut, running_mean=self.weights_br['b{}_0/norm2/mean'.format(block_index)],
                                running_var=self.weights_br['b{}_0/norm2/var'.format(block_index)],
                                weight=weights['b{}_0/norm2/weight'.format(block_index)],
                                bias=weights['b{}_0/norm2/bias'.format(block_index)], training=bn_training,momentum=0.1)
        x += shortcut
        x = self.relu(x, inplace=True)
        return x

    def conv_block(self, input_, weights, block_index, index_in_block, bn_training, droput_training):
        if self.use_conv_bias:
            x = F.conv2d(input=input_, weight=weights['b{}_{}/conv0/filter'.format(block_index, index_in_block)],
                         bias=weights['b{}_{}/conv0/bias'.format(block_index, index_in_block)], stride=1, padding=1)
        else:
            x = F.conv2d(input=input_, weight=weights['b{}_{}/conv0/filter'.format(block_index, index_in_block)],
                         bias=None, stride=1, padding=1)
        x = F.batch_norm(input=x, running_mean=self.weights_br['b{}_{}/norm0/mean'.format(block_index, index_in_block)],
                         running_var=self.weights_br['b{}_{}/norm0/var'.format(block_index, index_in_block)],
                         weight=weights['b{}_{}/norm0/weight'.format(block_index, index_in_block)],
                         bias=weights['b{}_{}/norm0/bias'.format(block_index, index_in_block)], training=bn_training,
                         momentum=0.1)
        x = self.relu(x, inplace=True)

        if self.use_dropout: x = F.dropout(input=x, p=self.dropout_p, training=droput_training)

        if self.use_conv_bias:
            x = F.conv2d(input=x, weight=weights['b{}_{}/conv1/filter'.format(block_index, index_in_block)],
                         bias=weights['b{}_{}/conv1/bias'.format(block_index, index_in_block)], stride=1, padding=1)
        else:
            x = F.conv2d(input=x, weight=weights['b{}_{}/conv1/filter'.format(block_index, index_in_block)],
                         bias=None, stride=1, padding=1)
        x = F.batch_norm(input=x, running_mean=self.weights_br['b{}_{}/norm1/mean'.format(block_index, index_in_block)],
                         running_var=self.weights_br['b{}_{}/norm1/var'.format(block_index, index_in_block)],
                         weight=weights['b{}_{}/norm1/weight'.format(block_index, index_in_block)],
                         bias=weights['b{}_{}/norm1/bias'.format(block_index, index_in_block)], training=bn_training,
                         momentum=0.1)
        x += input_
        x = self.relu(x, inplace=True)
        return x

    def fc_forward(self, input_, weights, bn_training):
        return F.linear(input=input_, weight=weights['fc/weight'], bias=weights['fc/bias'])

    def forward(self, x, weights=None, bn_training=True, dropout_training=True):
        if weights is None:
            weights_use = self.weights
        else:
            weights_use = {}
            for key_ in self.weights.keys():
                if key_ in weights.keys():
                    weights_use[key_] = weights[key_]
                else:
                    weights_use[key_] = self.weights[key_]

        x = self.conv_norm_activation(input_=x, weights=weights_use, bn_training=bn_training)

        for block_index in range(1, 4):
            x = self.expand_conv(input_=x, weights=weights_use, block_index=block_index, bn_training=bn_training,
                                 droput_training=dropout_training)
            for index_in_block in range(1, self.num_blocks):
                x = self.conv_block(input_=x, weights=weights_use, block_index=block_index,
                                    index_in_block=index_in_block, bn_training=bn_training,
                                    droput_training=dropout_training)

        x = nn.AvgPool2d(10, stride=1)(x)
        x = x.view(x.size(0), -1)
        x = self.fc_forward(input_=x, weights=weights_use, bn_training=bn_training)
        return x

    def zero_grad(self, weights=None):
        with torch.no_grad():
            if weights is None:
                for p in self.weights.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in weights.values():
                    if p.grad is not None:
                        p.grad.zero_()

            for t in self.task_lr.values():
                if t.grad is not None:
                    t.grad.zero_()

    def parameters(self):
        return self.weights.values()

    def named_parameters(self):
        return self.weights.items()

    def dict_parameters(self):
        return self.weights

    def parameters_by_names(self, names):
        tmp = list(filter(lambda x: x[0] in names, self.named_parameters()))
        return [x[1] for x in tmp]

    def parameters_inner_item(self, names):
        tmp = list(filter(lambda x: x[0] in names, self.named_parameters()))
        return tmp




















