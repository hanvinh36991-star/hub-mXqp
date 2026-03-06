"""
1、 理解rnn、lstm、gru的计算过程（面试用途），阅读官方文档 ：https://docs.pytorch.org/docs/2.4/nn.html#recurrent-layers
最终 使用rnn/ lstm / gru 分别代替原始lstm，实现05_LSTM文本分类.py，进行实验，对比精度
"""
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 数据处理阶段
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
# 抽取第一列 文本描述
texts = dataset[0].tolist()
# 抽取第二列 标签
string_labels = dataset[1].tolist()

#  set是对标签进行去重，enumerate是返回标签-索引对迭代器。
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 所有原始标签的索引列表，输出结果是一个整数列表
numerical_labels = [label_to_index[label] for label in string_labels]

# 生成词汇表，是将所有输入的文本对应按照 字-索引对，按出现顺序进行升序
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 将 词-索引 反转成 索引-词
index_to_char = {i: char for char, i in char_to_index.items()}

# 词汇表长度
vocab_size = len(char_to_index)

# 将每句text都标准化成40个字符，取长补短
max_len = 40


class CharLSTMDataset(Dataset):
    """
    自定义数据集，方便后续训练的时候读取数据
    """

    def __init__(self, texts, labels, char_to_index, max_len):
        """
        texts: 原始文本列表
        labels: 处理后的 原始文本对应的标签索引列, 是整数列表
        char_to_index: 处理后的 字-索引 对字典
        max_len: 将每一句不等长的原始文本，标准化成40个字符长度
        """
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        """
        返回原始输入文本的长度，代表待训练的数据集大小
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        目的：将原始文本 按词获取对应的索引，若少于40个，则补0，多于40个词，则舍弃
        返回：2个，1.对应那句文本的词索引列表【设定的长度max_len】，2.对应那句文本的索引值，返回的是Tensor类型
        """
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
# 这是将所有的待训练的数据集，分成32一组的迭代器，会返回 32*40这个维度的文本，以及对应的32*1的对应每个文本的索引[这个索引是对应的标签列表的索引]。
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)


# --- NEW LSTM Model Class ---
class LSTMClassifier(nn.Module):
    """
    LSTM 分类器模型
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        """
        vocab_size: 词汇表大小，待训练的词汇表的大小
        embedding_dim: 词向量嵌入层维度
        hidden_dim：隐藏层维度
        output_dim：输出层维度
        """
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)  # 全连接层

    def forward(self, x):
        """
        前向传播，当执行 model(x)，就是在执行这个函数
        """
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out


# --- NEW RNN Model Class ---
class RNNClassifier(nn.Module):
    """
    RNN 分类器模型
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        """
        vocab_size: 词汇表大小，待训练的词汇表的大小
        embedding_dim: 词向量嵌入层维度
        hidden_dim：隐藏层维度
        output_dim：输出层维度
        """
        super(RNNClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程， 可训练的
        self.RNN = nn.RNN(
            embedding_dim,  # 输入特征数
            hidden_dim,  # 隐藏状态特征数
            num_layers=2,  # RNN循环层数，例如，设置num_layers=2意味着将两个RNN堆叠在一起，形成堆叠RNN，第二个接收第一个RNN的输出并计算最终结果。默认为1
            nonlinearity='tanh',  # 使用的非线性函数，默认tanh,可选relu
            bias=True,  # 如果为False，则该层不使用偏置权重 b_ih 和 b_hh，默认True
            batch_first=True,  # True，顺序为: batch, seq, feature 若为false，则seq, batch, feature
            dropout=0,  # 如果非0，则在除最后一层之外的每个RNN层的输出上引入Dropout层，dropout概率等于dropout。默认值为0
            bidirectional=False  # 如果为True, 则成为双向RNN，默认值为False
        )  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)  # 全连接层

    def forward(self, x):
        """
        前向传播，当执行 model(x)，就是在执行这个函数
        """
        embedded = self.embedding(x)
        # RNN返回: output, hidden
        # output: [batch_size, seq_len, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        output, hidden = self.RNN(embedded)
        # 取最后一个时间步的隐藏状态
        out = self.fc(hidden[-1])
        return out


# --- NEW GRU Model Class ---
class GRUClassifier(nn.Module):
    """
    GRU 分类器模型
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        """
        vocab_size: 词汇表大小，待训练的词汇表的大小
        embedding_dim: 词向量嵌入层维度
        hidden_dim：隐藏层维度
        output_dim：输出层维度
        """
        super(GRUClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程， 可训练的
        self.GRU = nn.GRU(
            embedding_dim,  # 输入特征维度
            hidden_dim,  # 隐藏层特征维度
            num_layers=2,  # 循环层数，例如设置num_layers=2，意味着将两个GRU堆叠在一起，以形成堆叠的GRU，第二个GRU接收第一个GRU的输出并计算最终结果，默认值：1
            bias=True, # 如果为False，则该层不使用偏置权重 b_ih 和 b_hh，默认True
            batch_first=True,
            dropout=0, # 如果非0，则在除最后一层之外的每个GRU层的输出上引入Dropout层，dropout概率等于dropout。默认值为0
            bidirectional=False # 如果为True, 则成为双向GRU，默认值为False
        )  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)  # 全连接层

    def forward(self, x):
        """
        前向传播，当执行 model(x)，就是在执行这个函数
        """
        embedded = self.embedding(x)
        output, hidden = self.GRU(embedded)
        out = self.fc(hidden[-1])
        return out


embedding_dim = 64  # 为啥要将词向量维度设置成64，我也不明白
hidden_dim = 128  # 为啥将隐藏层维度设置成128，我也不明白
output_dim = len(label_to_index)  # 输出维度，为啥是词汇表的长度咧？

# model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)  # 初始化模型

# 模型训练与评估
models = {
    'KNN': RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
    'LSTM': LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
    'GRU': GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
}

count_loss = {}
for name, model in models.items():
    print(f"\n ---------正在评估模型：{name}---------")
    count_loss[name] = []
    criterion = nn.CrossEntropyLoss()  # 初始化损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 初始化优化器
    # 开始训练
    num_epochs = 4  # 控制将同一批训练数据，训练四轮
    for epoch in range(num_epochs):
        model.train()  # 将模型调整成训练模式
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()  # 清除模型参数的梯度
            outputs = model(inputs)  # 训练模型
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 计算梯度
            optimizer.step()  # 根据计算得到的梯度，更新参数
            running_loss += loss.item()
            if idx % 50 == 0:
                print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
        count_loss[name].append(running_loss)

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    """
    预测数据 进行相同的数据处理操作。
    """
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    # 开启评估模式
    model.eval()
    # 禁用梯度
    with torch.no_grad():
        output = model(input_tensor)

    # 在第一维度上找最大值（类别维度），返回两个参数，第一个参数是最大值的具体值，第二个返回最大值所在索引
    _, predicted_index = torch.max(output, 1)
    # 将单元素张量转换为Python标量, 前提：batch_size 必须为1，如果大于1，则为多个样本，则需要遍历
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}
new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")


# 画图对比
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决符号显示问题
plt.Figure(figsize=(10,6))

# 绘制每条折线
for model_name, values in count_loss.items():
    plt.plot(values, marker='o', label=model_name, linewidth=2)

# 添加标签和标题
plt.xlabel('迭代次数/时间点', fontsize=12)
plt.ylabel('数值', fontsize=12)
plt.title('KNN、LSTM和GRU性能比较', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)

# 添加网格
plt.grid(True, alpha=0.3)

# 显示图形
plt.tight_layout()
plt.show()