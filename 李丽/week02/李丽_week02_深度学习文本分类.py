import pandas as pd
import torch
import torch.nn as nn #nn 是 Neural Networks（神经网络） 的缩写
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. 数据加载与预处理
# 读取CSV数据集，使用制表符作为分隔符
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()  # 提取文本列
string_labels = dataset[1].tolist()  # 提取标签列

# 标签编码：将字符串标签转换为数值索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建字符词典：将文本中的每个字符映射为唯一的整数索引
char_to_index = {'<pad>': 0}  # 0用于填充
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 创建反向映射：索引 -> 字符
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)  # 词汇表大小

max_len = 40  # 设定文本最大长度，超过截断，不足填充

# 新增自定义数据集
# torch dataset：负责读取单个样本，提供__getitem__方法
# torch dataloader：负责将多个样本拼接为batch，提供多线程加载等功能

class CharBoWDataset(Dataset):
    """
    自定义数据集类，用于加载文本分类数据并转换为BoW向量。
    继承自 torch.utils.data.Dataset
    """
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)  # 将标签转换为Tensor
        self.char_to_index = char_to_index
        self.max_len = max_len # 40
        self.vocab_size = vocab_size # 词汇表大小，即字符总数
        # 在初始化时预先计算好所有样本的BoW向量，也可以在__getitem__中动态计算
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        """
        将所有文本转换为BoW (Bag of Words) 向量
        """
        tokenized_texts = []
        for text in self.texts:
            # 将字符转换为索引，截断或填充
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            # 创建全0向量，长度为词汇表大小
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:  # 忽略填充符
                    bow_vector[index] += 1  # 统计词频
            bow_vectors.append(bow_vector)
        # 将列表转换为Tensor堆叠
        return torch.stack(bow_vectors)

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.texts)

    def __getitem__(self, idx):
        """根据索引返回单个样本（特征和标签）"""
        return self.bow_vectors[idx], self.labels[idx]


# --- 以下部分由用户更新：升级后的 SimpleClassifier 类，支持动态配置层数 ---
class SimpleClassifier(nn.Module):
    """
    可配置层数和节点数的全连接神经网络分类器 (升级版)
    对应 Playground 中的 Hidden Layers (层数) 和 Neurons (节点数)
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        # hidden_dims: 一个列表，例如 [128, 64] 表示有两个隐藏层，节点数分别为 128 和 64
        super(SimpleClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 动态构建隐藏层：根据 hidden_dims 列表的长度和数值自动添加层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim)) # 线性层：负责维度变换
            layers.append(nn.ReLU())                  # 激活函数：引入非线性
            prev_dim = h_dim # 更新下一层的输入维度
            
        # 输出层：最后一层将维度变换为分类类别数
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 使用 Sequential 将所有层串联起来，像穿糖葫芦一样
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
# --- 更新结束 ---


# 实例化Dataset
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) 
# 实例化DataLoader，batch_size=32表示每次读取32个样本，shuffle=True表示打乱数据
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) 

# --- 以下部分由用户更新：封装训练函数，方便重复实验 ---
def train_model(model, dataloader, num_epochs=10, learning_rate=0.01):
    """
    通用的训练函数
    """
    criterion = nn.CrossEntropyLoss() # 损失函数
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) # 优化器
    
    loss_history = []
    
    print(f"开始训练模型结构: {model}")
    for epoch in range(num_epochs): 
        model.train() # 训练模式
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()       # 梯度清零
            outputs = model(inputs)     # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward()             # 反向传播
            optimizer.step()            # 更新参数
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        # 每 5 个 Epoch 打印一次日志
        if (epoch + 1) % 5 == 0:
             print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
             
    return loss_history
# --- 更新结束 ---

output_dim = len(label_to_index)

# 原始的训练代码已被上面的 train_model 替代，这里直接调用进行实验

# --- 实验 1: 单隐藏层，128个节点 (基准模型) ---
print("\n--- 实验 1: 单隐藏层 [128] ---")
model_1 = SimpleClassifier(vocab_size, [128], output_dim)
loss_1 = train_model(model_1, dataloader, num_epochs=50)


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    """
    使用训练好的模型对新文本进行分类预测
    """
    # 1. 文本预处理（同训练时）
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    # 2. 构建BoW向量 bag of words 
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    # 3. 增加Batch维度 (1, vocab_size)
    bow_vector = bow_vector.unsqueeze(0)

    # 4. 模型预测
    model.eval() # 设置为评估模式
    with torch.no_grad(): # 不计算梯度
        output = model(bow_vector)

    # 5. 获取预测结果
    _, predicted_index = torch.max(output, 1) # 获取最大概率的索引
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

# 新增一个模型调用
print("\n--- 实验 2: 双隐藏层 [128, 64] ---")
model_2=SimpleClassifier(vocab_size, [128,64], output_dim)
loss_2=train_model(model_2, dataloader, num_epochs=50)


# 测试模型
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model_2, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model_2, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
