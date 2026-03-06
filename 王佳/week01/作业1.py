import sys
import jieba
import sklearn
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import gensim
import peft
import transformers as tfs
import fastapi

version_info = sys.version_info

print(f"python当前版本：{version_info.major}.{version_info.minor}.{version_info.micro}")
print(f"jieba当前版本：{jieba.__version__}")
print(f"sklearn当前版本：{sklearn.__version__}")
print(f"pytorch当前版本：{torch.__version__}")
print(f"numpy当前版本：{np.__version__}")
print(f"pandas当前版本：{pd.__version__}")
print(f"matplotlib当前版本：{mpl.__version__}")
print(f"gensim当前版本：{gensim.__version__}")
print(f"peft当前版本：{peft.__version__}")
print(f"transformers当前版本：{tfs.__version__}")
print(f"fastapi当前版本：{fastapi.__version__}")