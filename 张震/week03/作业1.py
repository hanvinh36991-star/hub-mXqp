import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

dataset = pd.read_csv("../../Week01/00老师代码/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
index_to_label = {i: label for label, i in label_to_index.items()}
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
class ClassDataset(Dataset):
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


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        rnn_out, hidden = self.rnn(embedded)

        # batch size * output_dim
        out = self.fc(hidden.squeeze(0))
        return out


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        gru_out, hidden = self.gru(embedded)

        # batch size * output_dim
        out = self.fc(hidden.squeeze(0))
        return out


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0))
        return out

# --- Training and Prediction ---
class_dataset = ClassDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(class_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

criterion = nn.CrossEntropyLoss()

def run_model(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 6
    loss_all = []
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        watch_loss = []
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
            if idx % 50 == 0:
                print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(watch_loss):.4f}")
        loss_all = loss_all + watch_loss

    return loss_all

def class_text(new_text, model, criterion, char_to_index, max_len, index_to_label, label_true):
    # 评估模式
    model.eval()

    indices = [char_to_index.get(char, 0) for char in new_text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    label_tensor = torch.tensor(label_true, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        loss = criterion(output, label_tensor)

    _, predicted = torch.max(output, 1)
    predicted_index = predicted.item()
    predicted_label = index_to_label[predicted_index]
    return predicted_label, loss

# 读取测试集
test_dateset = pd.read_csv("test_dataset.csv", sep=",")
test_texts = test_dateset["query"]
test_labels = test_dateset["intent"]

def check_model(model, criterion, char_to_index, max_len, index_to_label):
    total = len(test_texts)
    check = []
    check_loss = []
    for i in range(total//100):
        correct, wrong = 0, 0
        running_loss = []
        for j in range(100):
            k = i * 100 + j
            label_true = test_labels[k]
            label_index = label_to_index[label_true]
            label_pred, loss = class_text(test_texts[k], model, criterion, char_to_index, max_len, index_to_label, label_index)
            running_loss.append(loss)
            if label_pred == label_true:
                correct += 1
            else:
                wrong += 1
        check.append(correct / (correct + wrong))
        check_loss.append(np.mean(running_loss))
    return check, check_loss


# rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# rnn_loss = run_model(rnn_model)
# rnn_check = check_model(rnn_model, char_to_index, max_len, index_to_label)
#
# gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# gru_loss = run_model(gru_model)
# gru_check = check_model(gru_model, char_to_index, max_len, index_to_label)

lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
lstm_loss = run_model(lstm_model)
lstm_check, lstm_loss_check = check_model(lstm_model, criterion, char_to_index, max_len, index_to_label)


"""
new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
"""

# 设置全局字体为支持中文的字体，例如使用“SimHei”
matplotlib.rcParams['font.family'] = 'SimHei'  # 或 'Microsoft YaHei' 等
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# plt.subplot(1, 3, 2)
plt.xlabel('Epoch')
plt.ylabel('Loss和准确率')
plt.title('测试集准确率曲线')
#plt.plot(rnn_loss, label='RNN LOSS', color='green')
# plt.plot(rnn_check, label='RNN 准确率', color='green')
# #plt.plot(gru_loss, label='GRU LOSS', color='blue')
# plt.plot(gru_check, label='GRU 准确率', color='blue')
plt.plot(lstm_loss_check, label='LSTM LOSS', color='red')
plt.plot(lstm_check, label='LSTM 准确率', color='m')
#plt.yscale('log')  # 使用对数坐标更清晰
plt.legend()
plt.grid(True)

plt.show()
