#结巴
import jieba
print("jieba安装成功")

# sklearn
import sklearn
print("sklearn安装成功")

# PyTorch
import torch

print(f"PyTorch版本：{torch.__version__}")
print(f"CUDA是否可用：{torch.cuda.is_available()}")

########################其他库#####################
import numpy
print("numpy 已安装，版本：", numpy.__version__)

import pandas
print("pandas 已安装，版本：", pandas.__version__)

import openpyxl
print("openpyxl 已安装，版本：", openpyxl.__version__)
