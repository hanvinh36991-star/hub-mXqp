# 导入必要的库
import pandas as pd  # 用于数据读取和处理
import torch         # PyTorch 主库
import torch.nn as nn # 神经网络模块
import torch.optim as optim # 优化器模块
from torch.utils.data import Dataset, DataLoader # 数据集和数据加载器

# ==========================
# 1. 数据读取和预处理
# ==========================
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None) # 读取csv文件，tab分隔，文件没有表头
texts = dataset[0].tolist()       # 第一列是文本数据
string_labels = dataset[1].tolist() # 第二列是文本标签
print(f"{set(string_labels)}")

# 将字符串标签映射成整数索引，例如 "cat" -> 0, "dog" -> 1
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建字符到索引的映射，用于字符级BOW编码
char_to_index = {'<pad>': 0} # <pad> 表示填充字符
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index) # 为新字符分配一个唯一索引

# 构建索引到字符的映射（反向映射）
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)  # 词表大小
max_len = 40                      # 文本截断或填充的最大长度

# ==========================
# 2. 构建自定义数据集
# ==========================
class CharBoWDataset(Dataset):
    """
    字符级 BOW 数据集，将文本转换为Bag-of-Words向量
    """
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)  # 标签转成tensor
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()        # 初始化时生成BOW向量

    def _create_bow_vectors(self):
        """
        将文本转换为BOW向量
        """
        tokenized_texts = []
        for text in self.texts:
            # 将每个字符转换为索引，并截断或填充到max_len
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized)) # 填充
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size) # 初始化BOW向量
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1  # 对出现的字符计数
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)  # 转换为tensor

    def __len__(self):
        return len(self.texts)  # 样本数

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx] # 返回BOW向量和标签

# ==========================
# 3. 构建简单神经网络分类器
# ==========================
class SimpleClassifier(nn.Module):
    """
    简单全连接分类器
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层全连接
        self.relu = nn.ReLU()                        # 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 输出层

    def forward(self, x):
        # 前向传播
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ==========================
# 4. 数据集和数据加载器
# ==========================
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 批量读取数据

# ==========================
# 5. 模型、损失函数和优化器
# ==========================
hidden_dim = 1024
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, output_dim) # 输入维度=vocab_size, 输出维度=类别数
criterion = nn.CrossEntropyLoss() # 多分类交叉熵损失，内部自带softmax
optimizer = optim.SGD(model.parameters(), lr=0.01) # 随机梯度下降优化器

# ==========================
# 6. 训练循环
# ==========================
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()        # 清空梯度
        outputs = model(inputs)      # 前向传播
        loss = criterion(outputs, labels) # 计算损失
        loss.backward()              # 反向传播计算梯度
        optimizer.step()             # 更新参数
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

# ==========================
# 7. 定义文本分类函数
# ==========================
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    """
    将输入文本转换为BOW向量并预测类别
    """
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized)) # 填充

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1
    bow_vector = bow_vector.unsqueeze(0) # 增加batch维度

    model.eval() # 切换到推理模式
    with torch.no_grad(): # 不计算梯度
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1) # 取概率最大索引
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

# 构建索引到标签的映射
index_to_label = {i: label for label, i in label_to_index.items()}

# ==========================
# 8. 测试文本预测
# ==========================
def pre_text_test(text, model, char_to_index, vocab_size, max_len, index_to_label):
    new_text = text
    predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

print(f"当前隐藏层数:{hidden_dim} num_epochs = {num_epochs}")
pre_text_test("帮我导航到北京", model, char_to_index, vocab_size, max_len, index_to_label)
pre_text_test("查询明天北京的天气", model, char_to_index, vocab_size, max_len, index_to_label)
pre_text_test("帮我购买周杰伦的演唱会门票", model, char_to_index, vocab_size, max_len, index_to_label)
pre_text_test("我想听邓紫棋的歌", model, char_to_index, vocab_size, max_len, index_to_label)
pre_text_test("我想看周星驰的电影", model, char_to_index, vocab_size, max_len, index_to_label)
pre_text_test("帮我导航到北京", model, char_to_index, vocab_size, max_len, index_to_label)
pre_text_test("我想看钢铁侠3", model, char_to_index, vocab_size, max_len, index_to_label)
pre_text_test("帮我预定明天10点的会议", model, char_to_index, vocab_size, max_len, index_to_label)
pre_text_test("上海明天的天气怎么样", model, char_to_index, vocab_size, max_len, index_to_label)
pre_text_test("空调温度26度", model, char_to_index, vocab_size, max_len, index_to_label)