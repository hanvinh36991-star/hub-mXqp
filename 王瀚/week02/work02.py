# ================= 对比实验 =================
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random, numpy as np

# 设置随机数种子
random_seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # CUDA 卷积确定
    torch.backends.cudnn.benchmark = False
set_seed(random_seed)

# ------------ 加载数据和预处理 --------------
dataset = pd.read_csv("E:\AIProject\Week01\dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40
output_dim = len(label_to_index)

class CharBoWDataset(Dataset):# 自定义数据集 custom dataset，对不同的数据集写法不同
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size): # 传入需要建模的文本
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self): # 取长补短
        # pad and crop
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        # 词频编码
        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

# ------------ 模型超参数 ------------------
hidden_dims = [112, 176, 240] # [168, 176, 184]
num_layerss = [1, 2, 3]
num_epochs  = 80
batch_size  = 32
lr          = 0.03 # 0.01
dropout     = 0.1

# ------------ 数据拆分（固定） -------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, numerical_labels, test_size=0.15, random_state=random_seed,
        stratify=numerical_labels)

train_dataset = CharBoWDataset(train_texts, train_labels, char_to_index, max_len, vocab_size)
val_dataset   = CharBoWDataset(val_texts,   val_labels,   char_to_index, max_len, vocab_size)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# ------------ 可配置层数的分类器 -------------
class FlexibleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        layers = []

        dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        for i in range(1, len(dims) - 1):
            layers += [nn.Linear(dims[i - 1], dims[i])]
            layers += [nn.BatchNorm1d(dims[i])]  # ← 归一化
            layers += [nn.SiLU()] #①nn.ReLU()   ② nn.ELU(1.0)   ③ nn.SiLU()
            layers += [nn.Dropout(dropout)]  # ← 随机失活
        layers += [nn.Linear(dims[-2], dims[-1])]  # 输出层不加 BN/Dropout
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------ 单组实验函数 -------------
def run_one_exp(hidden_dim, num_layers):
    model = FlexibleClassifier(vocab_size, hidden_dim, output_dim, num_layers, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    best_val_acc, best_val_loss = 0.0, float('inf')
    # history：记录每个 epoch 的 train & val 指标
    history = {'train_acc':[], 'train_loss':[], 'val_acc':[], 'val_loss':[]}

    for epoch in range(num_epochs):
        # ---------- 训练 ----------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss   += loss.item() * y.size(0)
            train_correct += out.argmax(1).eq(y).sum().item()
            train_total   += y.size(0)
        train_acc = train_correct / train_total
        train_loss = train_loss / train_total

        # ---------- 验证 ----------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                val_loss += criterion(out, y).item() * y.size(0)
                val_correct += out.argmax(1).eq(y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total
        val_loss = val_loss / val_total

        # 记录历史
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        # 更新最佳
        best_val_acc  = max(best_val_acc, val_acc)
        best_val_loss = min(best_val_loss, val_loss)

    return best_val_acc, best_val_loss, history

# ------------ 收集数据 -------------
results = []
histories = []
for hd, nl in tqdm(itertools.product(hidden_dims, num_layerss)):
    acc, loss, history = run_one_exp(hd, nl)
    results.append({'hidden_dim':hd, 'num_layers':nl,
                    'best_val_acc':acc, 'best_val_loss':loss})
    histories.append({'hidden_dim':hd, 'num_layers':nl, 'history':history})
    print(f'hd={hd}  nl={nl}  best_acc={acc:.4f}  best_loss={loss:.4f}')

# ------------ 可视化 --------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示为方块

df = pd.DataFrame(results)

# 1. 准确率热力图
pivot_acc = df.pivot(index='num_layers', columns='hidden_dim', values='best_val_acc')
plt.figure(figsize=(4, 3))
sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='Blues', cbar_kws={'label': 'Accuracy'})
plt.title('Validation Accuracy (layer × nodes)')
plt.ylabel('num_layers'); plt.xlabel('hidden_dim')
plt.tight_layout()
plt.savefig('heatmap_acc.png', dpi=120)
plt.show()

# 2. 损失热力图
pivot_loss = df.pivot(index='num_layers', columns='hidden_dim', values='best_val_loss')
plt.figure(figsize=(4, 3))
sns.heatmap(pivot_loss, annot=True, fmt='.3f', cmap='Reds_r', cbar_kws={'label': 'Loss'})
plt.title('Validation Loss (layer × nodes)')
plt.ylabel('num_layers'); plt.xlabel('hidden_dim')
plt.tight_layout()
plt.savefig('heatmap_loss.png', dpi=120)
plt.show()

# 3. 双指标柱状图
df['setting'] = df['num_layers'].astype(str) + '层×' + df['hidden_dim'].astype(str) + '节点'
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# accuracy bar
sns.barplot(x='setting', y='best_val_acc', data=df, ax=ax[0], palette='muted')
ax[0].set_ylabel('Best Val Accuracy')
ax[0].set_title('Accuracy vs Architecture')
ax[0].tick_params(axis='x', rotation=45)

# loss bar
sns.barplot(x='setting', y='best_val_loss', data=df, ax=ax[1], palette='rocket_r')
ax[1].set_ylabel('Best Val Loss')
ax[1].set_title('Loss vs Architecture')
ax[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('bar_acc_loss.png', dpi=120)
plt.show()

# 4. 训练vs验证曲线
import math
n_arch = len(histories)
cols = 3                    # 每行 3 张
rows = math.ceil(n_arch / cols)

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True, sharey=True)
axes = axes.flatten() if rows*cols > 1 else [axes]

for ax, h in zip(axes, histories):
    hist = h['history']
    label = f"{h['hidden_dim']}hid×{h['num_layers']}lay"
    ax.plot(hist['train_acc'], label='train', color='#1f77b4', lw=2)
    ax.plot(hist['val_acc'],   label='val',   color='#ff7f0e', lw=2, ls='--')
    ax.set_title(label)
    ax.legend()
    ax.set_ylim(0, 1)

# 隐藏多余子图
for ax in axes[n_arch:]:
    ax.axis('off')

plt.suptitle('Train vs Val Accuracy  (one panel per architecture)')
plt.tight_layout()
plt.savefig('acc_subplots.png', dpi=150)
plt.show()


# ------------ 调参过程 --------------
# 1. 隐藏层优化
# 开始时设置三个隐藏层，但是发现一层->二层->三层的效果越来越差，无论epoch加到多少都
# 改变不了。在大模型的帮助下，我在每一层后面加了BatchNorm1d和Dropout，发现三层的
# 准确率有明显的提升
# 如果继续按照单独的层去优化，可能需要的时间成本比较多，有点难控制变量，另外就是模型
# 已经足够好了，我感觉没有必要

# 2. 节点数
# 我使用二分的方式，最终找到一个相对比较好的区间（[168, 176, 184]），比这个区间偏大或偏小的效果都会有点差，
# 节点数过大的时候收敛的会比较慢，我认为仅针对这个任务，通过调参已经很难再有特别大的提升了，
# 所以没有必要去花大量时间继续去增加节点数，想要再有明显提升只能从数据上面下工夫
# 另外就是，不同层的最佳区间可能不一样，这个我也没有去深究，我只是看不同的结果后有这种感觉

# 3. epoch
# 从准确率曲线可以看出，其实这个任务收敛的很快，epoch继续提升带来的准确率提升并不明显，
# 当然还是有提升的，然后就是针对节点数比较多的情况，需要设置比较多的epoch来让模型收敛

# 4. 学习率
# 在调参的最后阶段，继续降低学习率有可能进一步提升准确率，不过不能降的太多，按阶梯形式往下调就行了，
# 大概从0.03调到0.01

# 5. 激活函数
# 尝试了几种不同的激活函数，最终选择了SiLU这个函数，效果相对好一些，不过这个因任务而异

# 6. 随机性
# 模型的随机性比较大，设置随机种子后，波动稍微小了一些，不过还是会有波动，一般是在0.92~0.93范围内波动，
# 极端情况下可能会过好（接近0.94）或者略差（0.91）
