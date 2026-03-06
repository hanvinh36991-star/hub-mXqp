"""
1、调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
"""
import numpy as np
from matplotlib import pyplot as plt

"""
1. 自定义数据集，模型的定义，模型的训练
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签编码，将字符串形式的分类标签转换为数值形式。
# enumerate(...)：为每个唯一标签分配索引（0, 1, 2, ...），创建一个映射字典，标签-索引对
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 遍历 string_labels 中的每个原始标签，使用 label_to_index 字典查找对应的数值索引，构建新的数值标签列表
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建一个字符级别的词汇表，用于将字符映射到数值索引，字符为key，索引为value
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 交换，把索引和字符交换过来，索引作为Key，字符作为value
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    """
    自定义数据集（custom dataset），继承自pytoche的dataset，然后做一个自定义。每个数据集写法不同
    用途：
    1.初始化的时候，传入需要建模的文本
    2.使用len，调用__len__，返回数据集的长度
    3.做索引，调用__getitem__
    """

    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        """
        texts: 是语句列表
        labels: 是标签列表
        char_to_index: 是字符-索引字典，key是词汇，value是索引
        max_len: 设置了取长补短，字符超过40截断，字符不足40补齐
        vocab_size: char_to_index 的长度
        """
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        # 取长补短
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


# 3层网络结构
class SimpleClassifier(nn.Module):
    """
    创建模型
    """

    def __init__(self, input_dim, hidden_dim, output_dim):  # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 4层网络结构
class SimpleClassifier_4(nn.Module):
    """
    创建模型
    """

    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):  # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier_4, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.elu = nn.ELU()
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


# torch dataset是读取单个样本
# torch dataloader 多个样本拼接为batch
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)  # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)  # 读取批量数据集 -》 batch数据

hidden_dim = 128
hidden_dim_1 = 256
hidden_dim_2 = 128
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, output_dim)  # 维度和精度有什么关系？
model_2 = SimpleClassifier_4(vocab_size, hidden_dim_1, hidden_dim_2, output_dim)
criterion = nn.CrossEntropyLoss()  # 损失函数 内部自带激活函数，softmax
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer_2 = optim.SGD(model_2.parameters(), lr=0.01)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次
loss_nums = []
num_epochs = 10
for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    loss_nums.append(running_loss / len(dataloader))
loss_nums_2 = []
num_epochs = 10
for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model_2.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer_2.zero_grad()
        outputs = model_2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_2.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    loss_nums_2.append(running_loss / len(dataloader))


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    """
    text: 待预测的字符串
    char_to_index: 是训练集，字符-索引字典，key是词汇，value是索引
    vocab_size: char_to_index 的长度
    max_len: max_len = 40，定义的训练数据，最大长度
    index_to_label
    """
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label



index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model_2, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

# 画图对比
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决符号显示问题
categories = [0]
for _ in range(len(loss_nums) - 1):
    categories.append(len(categories))
# 创建图形
plt.figure(figsize=(10, 6))

# 绘制柱状图
x = np.arange(len(categories))
width = 0.35

bars1 = plt.bar(x - width / 2, loss_nums, width, label='3层网络模型loss变化', alpha=0.7, color='cornflowerblue')
bars2 = plt.bar(x + width / 2, loss_nums_2, width, label='4层网络模型loss变化', alpha=0.7, color='salmon')

# 添加标签和标题
plt.xlabel('类别')
plt.ylabel('数值')
plt.title('两个列表的柱状图对比')
plt.xticks(x, categories)
plt.legend()

# 调整布局
plt.ylim(0, max(max(loss_nums), max(loss_nums_2)) * 1.2)
plt.tight_layout()
plt.show()
