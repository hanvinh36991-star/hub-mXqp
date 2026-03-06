import warnings
import re
from functools import reduce
import shutil
import time
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="jieba",
    message="pkg_resources is deprecated as an API"
)

from importlib_metadata import version

import pandas as pd
import jieba
import matplotlib
import torch
import gensim
import peft
import numpy as np
import transformers
import fastapi
from mypy.version import __version__ as mypy_version
import sklearn

def main():
    # 打印各库版本号
    print(f"pandas: {pd.__version__}")
    # jieba用__jieba_version__获取版本
    print(f"jieba: {version('jieba')}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"torch (PyTorch): {torch.__version__}")
    print(f"gensim: {gensim.__version__}")
    print(f"peft: {peft.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"fastapi: {fastapi.__version__}")
    # mypy使用导入的版本变量
    print(f"mypy: {mypy_version}")
    print(f"scikit-learn (sklearn): {sklearn.__version__}")
    print(f"np.version: {np.version.version}")

if __name__ == "__main__":
    main()
