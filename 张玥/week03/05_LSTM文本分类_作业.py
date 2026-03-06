import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("../../Week01/dataset.csv", sep="\t", header=None)
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

# max length 最大输入的文本长度
max_len = 40

# 自定义数据集 - 》 为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 统一的写法，底层pytorch 深度学习 / 大模型
class CharLSTMDataset(Dataset):
    # 初始化
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts # 文本输入
        self.labels = torch.tensor(labels, dtype=torch.long) # 文本对应的标签
        self.char_to_index = char_to_index # 字符到索引的映射关系
        self.max_len = max_len # 文本最大输入长度

    # 返回数据集样本个数
    def __len__(self):
        return len(self.texts)

    # 获取当个样本
    def __getitem__(self, idx):
        text = self.texts[idx]
        # pad and crop
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# a = CharLSTMDataset()
# len(a) -> a.__len__
# a[0] -> a.__getitem__


# --- NEW LSTM Model Class ---
'''
修改原有LSTMClassifier类，
使其支持RNN和GRU和LSTM三种类型的循环神经网络。
通过在初始化时传入rnn_type参数来选择使用哪种类型
'''
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type="lstm"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if rnn_type == "rnn":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Unsupported rnn type")

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.rnn_type = rnn_type

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)

        if self.rnn_type == "lstm":
            hidden_state = hidden[0]  # (h_n, c_n)
        else:
            hidden_state = hidden     # RNN / GRU

        out = self.fc(hidden_state.squeeze(0))
        return out


# --- Training and Prediction ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

# model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
'''
初始化RNNClassifier模型时，通过rnn_type参数选择使用GRU作为循环神经网络的类型。
'''
rnn_type = "lstm"  # "rnn" / "lstm" / "gru"
model = RNNClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    rnn_type= rnn_type
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        '''
        1. 清零梯度
        2. 前向传播
        3. 计算损失
        4. 反向传播
        5. 优化器更新参数
        6. 统计损失
        '''
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

# 对比精度
def evaluate(model, dataloader):
    # 1) 切到评估模式：关闭 Dropout、使用 BatchNorm 的推理统计等（当前模型没有BN/Dropout，但这是标准写法）
    model.eval()

    correct = 0  # 2) 统计预测正确的样本数
    total = 0    # 3) 统计总样本数

    # 4) 评估不需要反向传播，关闭梯度计算：更快、更省显存/内存
    with torch.no_grad():
        # dataloader 每次拿到一个 batch
        # inputs: [batch_size, seq_len]，例如 [32, 40]，是字符id序列
        # labels: [batch_size]，例如 [32]，是类别id（0..C-1）
        for inputs, labels in dataloader:
            # 5) 前向推理：outputs 是分类“分数”（logits）
            # outputs: [batch_size, num_classes]，例如 [32, C]
            outputs = model(inputs)

            # 6) 取每个样本在 num_classes 维度上最大的分数对应的类别id
            # torch.max(outputs, 1) 表示沿着 dim=1（类别维度）取最大值
            # 返回 (max_values, max_indices)
            # predicted: [batch_size]，每个元素是预测类别id
            _, predicted = torch.max(outputs, 1)

            # 7) 本 batch 的样本数
            total += labels.size(0)

            # 8) predicted == labels 得到一个布尔向量 [batch_size]
            # sum() 统计 True 的个数（预测正确的样本数）
            # item() 转成 Python 数值
            correct += (predicted == labels).sum().item()

    # 9) accuracy = 正确数 / 总数
    return correct / total


acc = evaluate(model, dataloader)
print(f"模型类型: {rnn_type}, 准确率: {acc:.4f}")

# 结果记录：模型类型: gru, 准确率: 0.9546
# 结果记录：模型类型: rnn, 准确率: 0.1087  实际上接近随机猜测，是梯度消失的典型表现
# 结果记录：模型类型: lstm, 准确率: 0.8797

'''
结论：在字符级文本分类任务中，普通 RNN 的训练 loss 长期维持在约 2.3（接近 ln(类别数)），
表明模型输出接近均匀分布，几乎未学习到有效特征；预测结果塌缩到单一类别，准确率接近随机猜测。
其根本原因在于 RNN 缺乏门控机制，在较长序列上容易出现梯度消失，难以捕捉关键上下文信息。
相比之下，LSTM 和 GRU 通过门控机制显著改善了信息传递和梯度流动，其中 GRU 结构更简洁，在有限训练轮次下更容易收敛，表现最佳。
'''
