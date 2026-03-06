"""
1、调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 加载数据集并提取文本和标签
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels= dataset[1].tolist()

# 将字符串标签转换为数值索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建字符到索引的映射表
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40

class CharBoWDataset(Dataset):
    """
    字符级别词袋模型数据集类

    Args:
        texts: 文本列表
        labels: 标签列表
        char_to_index: 字符到索引的映射字典
        max_len: 最大文本长度
        vocab_size: 词汇表大小
    """
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        """
        创建词袋向量

        Returns:
            torch.Tensor: 词袋向量张量
        """
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index !=0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

class SimpleClassifier(nn.Module):
    """
    简单的文本分类器模型

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            torch.Tensor: 模型输出张量
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class DeepSimpleClassifier(nn.Module):
    """
    深度文本分类模型（3层）
    Args:
        input_dim: 输入维度
        hidden_dim1: 第一个隐藏层维度
        hidden_dim2: 第二个隐藏维度
        output_dim: 输出维度
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(DeepSimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# 创建数据集和数据加载器
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数和优化器
hidden_dim = 128
output_dim = len(label_to_index)

# 实验1：原模型（2层，128节点）
print("=== 实验1：原模型（2层，128节点） ===")
model = SimpleClassifier(vocab_size, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
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
            print(f"Batch 个数 {idx}, 当前Batch Loss：{loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss：{running_loss / len(dataloader):.4f}")

print("\n=== 实验2：深度模型（3层，256->128节点） ===")
hidden_dim1 = 256
hidden_dim2 = 128
# 实验2：更深的模型（3层，256->128节点）
model2 = DeepSimpleClassifier(vocab_size, hidden_dim1, hidden_dim2, output_dim)
optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

for epoch in range(num_epochs):
    model2.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer2.zero_grad()
        outputs = model2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {idx}, Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss：{running_loss / len(dataloader):.4f}")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    """
    对输入文本进行分类预测

    Args:
        text: 待分类的文本
        model: 训练好的模型
        char_to_index: 字符到索引的映射字典
        vocab_size: 词汇表大小
        max_len: 最大文本长度
        index_to_label: 索引到标签的映射字典

    Returns:
        str: 预测的类别标签
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

# 构建索引到标签的映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试模型预测功能
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 ‘{new_text}’ 预测为：‘{predicted_class}’")

new_text_2 = "查询明天北京的天气"
predicted_class2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 ‘{new_text_2}’ 预测为：‘{predicted_class2}")
