# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
#
# # 定义你的神经网络模型
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         # 在此处定义你的神经网络结构
#
#     def forward(self, x):
#         # 在此处定义前向传播逻辑
#         return x
#
# # 创建模型实例并移动到GPU设备上
# model = MyModel().cuda()
#
#
#
# if torch.cuda.device_count() > 1:
#     print("使用{}块GPU进行并行训练".format(torch.cuda.device_count()))
#     # model = nn.DataParallel(model)

import sys
import pandas as pd

print("Python version:", sys.version)
print("Pandas version:", pd.__version__)
