import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from graph.tools import get_adjacency_matrix


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class TGCModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super(TGCModule, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class DownModule(nn.Module):
    def __init__(self):
        super(DownModule, self).__init__()
        self.down = lambda x: x

    def forward(self, x):
        return self.down(x)


class DTGCN(nn.Module):
    def __init__(self, out_channels, kernel_size, stride):
        super(DTGCN, self).__init__()
        self.tcms1 = nn.ModuleList()
        self.tcms2 = nn.ModuleList()

        self.tcms1.append(TGCModule(out_channels, out_channels*2, kernel_size=5, stride=stride))
        self.tcms1.append(TGCModule(out_channels, out_channels, kernel_size=7, stride=stride))
        self.tcms1.append(TGCModule(out_channels, out_channels*2, kernel_size=9, stride=stride))
        self.tcms2.append(TGCModule(out_channels*2, out_channels, kernel_size=9, stride=stride))
        self.tcms2.append(DownModule())
        self.tcms2.append(TGCModule(out_channels*2, out_channels, kernel_size=5, stride=stride))

    def forward(self, x):
        y = None
        for tcm1, tcm2 in zip(self.tcms1, self.tcms2):
            y_ = tcm2(tcm1(x))
            y = y_ + y if y is not None else y_
        return y


class STCAttention(nn.Module):
    def __init__(self, out_channels, num_jpts):
        super(STCAttention, self).__init__()

        # spatial attention
        ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)

        # temporal attention
        self.conv_ta = nn.Conv1d(out_channels, 1, 7, padding=3)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)

        # channel attention
        rr = 2
        self.fc1c = nn.Linear(out_channels, out_channels // rr)
        self.fc2c = nn.Linear(out_channels // rr, out_channels)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # spatial attention
        se = x.mean(-2)  # N C V
        se1 = self.sigmoid(self.conv_sa(se))
        y = x * se1.unsqueeze(-2) + x

        # temporal attention
        se = y.mean(-1)
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-1) + y

        # channel attention
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        return y


class MultiAdaptiveGCM(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, attention=True):
        super(MultiAdaptiveGCM, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        num_jpts = A.shape[-1]
        # initialize parameters for edge importance weighting
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))


        if attention:
            self.attention = STCAttention(out_channels, num_jpts)
        else:
            self.attention = lambda x: x

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        A = self.PA
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A[i] + A1 * self.alpha
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        y = self.bn(y)
        y = self.attention(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class STMultiAdaptiveGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, attention=True):
        super(STMultiAdaptiveGCN, self).__init__()
        # self.DTGCN = DTGCN(out_channels, kernel_size=7, stride=stride)
        self.magcm = MultiAdaptiveGCM(in_channels, out_channels, A, attention=attention)

        self.relu = nn.ReLU(inplace=True)

        self.tcms1 = nn.ModuleList()
        self.tcms2 = nn.ModuleList()

        self.tcms1.append(TGCModule(out_channels, out_channels*2, kernel_size=5, stride=stride))
        self.tcms1.append(TGCModule(out_channels, out_channels, kernel_size=7, stride=stride))
        self.tcms1.append(TGCModule(out_channels, out_channels*2, kernel_size=9, stride=stride))
        self.tcms2.append(TGCModule(out_channels*2, out_channels, kernel_size=9, stride=stride))
        self.tcms2.append(DownModule())
        self.tcms2.append(TGCModule(out_channels*2, out_channels, kernel_size=5, stride=stride))

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = TGCModule(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x_ = self.magcm(x)
        # y = self.DTGCN(x_)
        y = None
        for tcm1, tcm2 in zip(self.tcms1, self.tcms2):
            y_ = tcm2(tcm1(x_))
            y = y_ + y if y is not None else y_
        y = self.relu(y + self.residual(x))

        return y


class Model(nn.Module):
    def __init__(self,  in_channels, drop_out, adj_filename, id_filename, num_of_vertices, num_for_predict,
                 num_of_hours,num_of_days,num_of_weeks,):
        super(Model, self).__init__()
        A = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

        # num_pre = bool(num_of_hours) + bool(num_of_days) + bool(num_of_weeks)
        #self.data_bn = nn.BatchNorm1d(num_pre * num_of_vertices * 3)

        self.l1 = STMultiAdaptiveGCN(in_channels, 64, A, residual=False)
        self.l2 = STMultiAdaptiveGCN(64, 64, A,)
        self.l3 = STMultiAdaptiveGCN(64, 64, A,)

        self.final_conv = nn.Conv2d(int(num_for_predict/num_of_hours), num_for_predict, kernel_size=(1, 64))

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):

        N, M, T, V, C = x.size()  # (B, M, T, V, C)
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(N, M * V * C, T)
        #x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        # N*M,C,T,V
        x = self.drop_out(x).permute(0, 2, 3, 1).contiguous()
        x = self.final_conv(x).mean(-1)
        x = x.view(N, M, T, V).mean(1)

        return x
