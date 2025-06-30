import torch
import torch.nn as nn
# from einops import rearrange, repeat
import math
import torch.nn.functional as F

import numpy as np

__all__ = ['DRPCANet']

class DRPCANet(nn.Module):
    def __init__(self, stage_num=6, slayers=6, llayers=3, mlayers=5, channel=32, mode='train'):
        super(DRPCANet, self).__init__()
        self.stage_num = stage_num
        self.decos = nn.ModuleList()
        self.mode = mode
        for _ in range(stage_num):
            self.decos.append(DecompositionModule(slayers=slayers, llayers=llayers,
                                                  mlayers=mlayers, channel=channel))
        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, D):
        T = torch.zeros(D.shape).to(D.device)
        for i in range(self.stage_num):
            D, T = self.decos[i](D, T)
        if self.mode == 'train':
            return D,T
        else:
            return T

class DecompositionModule(object):
    pass


class DecompositionModule(nn.Module):
    def __init__(self, slayers=6, llayers=3, mlayers=5, channel=32):
        super(DecompositionModule, self).__init__()
        self.lowrank = LowrankModule(channel=channel, layers=llayers)
        self.sparse = DynamicSparseModule(channel=channel, layers=slayers)
        self.merge = DynamicResidualMergeModule(channel=channel, layers=mlayers)

    def forward(self, D, T):
        B = self.lowrank(D, T)
        T = self.sparse(D, B, T)
        D = self.merge(B, T)
        return D, T


class LowrankModule(nn.Module):
    def __init__(self, channel=32, layers=3):
        super(LowrankModule, self).__init__()

        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        #self.relu = nn.ReLU()
        #self.gamma = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

    def forward(self, D, T):
        x = D - T
        B = x + self.convs(x)
        return B


class DynamicSparseModule(nn.Module):
    def __init__(self, channel=32, layers=6):
        super(DynamicSparseModule, self).__init__()
        convs = [
            nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True)
        ]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)

        # 动态生成 gamma 的模块
        self.gamma_generator = parametergenerator(midchannel=3)
        # 动态生成 epsilon 的模块
        self.epsilon_generator = parametergenerator(midchannel=3)

    def forward(self, D, B, T):
        # 动态生成 gamma
        gamma = self.gamma_generator(T)  # 根据输入特征动态生成 gamma
        # 动态生成 epsilon
        x = gamma * T + (1 - gamma) * (D - B)
        epsilon = self.epsilon_generator(x)
        # 使用动态 epsilon 调整更新
        T = x - epsilon * self.convs(x)
        return T

class parametergenerator(nn.Module):
    def __init__(self, midchannel=3):
        super(parametergenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，提取全局特征
            nn.Conv2d(1, midchannel, kernel_size=1, bias=True),  # 卷积降维
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(midchannel, 1, kernel_size=1, bias=True),  # 卷积恢复维度
            nn.Sigmoid()  
        )
    def forward(self, x):
        return self.generator(x)

class DynamicResidualMergeModule(nn.Module):
    def __init__(self, channel=32, layers=5):
        super(DynamicResidualMergeModule, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=1),
                 nn.BatchNorm2d(channel),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        convs.append(DynamicResidualGroup(default_conv, channel, kernel_size = 3, reduction=16, n_resblocks=layers))
        convs.append(nn.Conv2d(channel, 1, kernel_size=1))
        self.mapping = nn.Sequential(*convs)

    def forward(self, B, T):
        x = B + T
        D = self.mapping(x)
        return D   

class ChannelAttention(nn.Module):
    def __init__(self, in_planes=32, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class DynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, kernel_size**2, kernel_size=1)  # [B, k*k, 1, 1]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 每个样本生成一个动态卷积核 [B, k*k, 1, 1] → [B, 1, k, k]
        kernels = self.kernel_generator(x).view(B, 1, self.kernel_size, self.kernel_size)
        # 2. 对每个样本取通道平均 [B, 1, H, W]
        x_mean = x.mean(dim=1, keepdim=True)
        # 3. reshape 成 grouped convolution 所需格式
        x_mean = x_mean.view(1, B, H, W)  # → [1, B, H, W]
        kernels = kernels.view(B, 1, self.kernel_size, self.kernel_size)  # [B, 1, k, k]
        # 4. 执行 grouped convolution，每个 kernel 只作用于对应的样本
        att = F.conv2d(
            x_mean,
            weight=kernels,
            padding=self.kernel_size // 2,
            groups=B
        )
        # 5. reshape 回原格式 + sigmoid
        att = att.view(B, 1, H, W)
        att = self.sigmoid(att)
        # 6. 应用注意力图
        return x * att

#未用分组卷积实现方法
# class DynamicSpatialAttention(nn.Module):
#     def __init__(self, in_channels=32, kernel_size=3):
#         super().__init__()
#         self.kernel_generator = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels, kernel_size=1),  # 生成动态卷积参数
#             nn.ReLU(),
#             nn.Conv2d(in_channels, kernel_size**2, kernel_size=1)  # 输出动态卷积核的权重
#         )
#         self.kernel_size = kernel_size
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # 生成动态卷积核权重
#         kernel_weights = self.kernel_generator(x)  # [B, kernel_size^2, 1, 1]
#         kernel_weights = kernel_weights.view(B, 1, self.kernel_size, self.kernel_size)  # [B, 1, k, k]
#         x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
#         att = []
#         for i in range(B):
#             att.append(F.conv2d(
#                 x_mean[i:i+1],  # 单个样本的输入
#                 kernel_weights[i:i+1],  # 对应样本的动态权重
#                 padding=self.kernel_size // 2
#             ))
#         att = torch.cat(att, dim=0)  # 合并所有样本的输出 [B, 1, H, W]
#         att = self.sigmoid(att)
#         return x * att

# Residual Channel Attention Block (RCAB)
class RCSAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCSAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(ChannelAttention())
        modules_body.append(DynamicSpatialAttention())
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

# Residual Group (RG)
class DynamicResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(DynamicResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCSAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=True, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)