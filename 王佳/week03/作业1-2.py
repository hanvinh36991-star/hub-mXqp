"""
作业1：
2.使用lstm ，使用rnn/ lstm / gru 分别代替原始lstm，进行实验，对比精度
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
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
class TextDataset(Dataset):
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

# GRU 模型
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
        gru_out, hidden_state = self.gru(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0))
        return out

# RNN 模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden_state = self.rnn(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm  = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# 训练模型
def train_model(model, dataloader, criterion, optimizer, num_epochs=4):
    start_time = time.time()
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/ {num_epochs}], Loss：{avg_loss:.4f}")

    training_time = time.time() - start_time
    return training_time

# 测试(评估)模型
def evaluate_model(model, test_texts, char_to_index, max_len, index_to_label):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for text in test_texts:
            indices = [char_to_index.get(char, 0) for char in text[:max_len]]
            indices += [0] * (max_len - len(indices))
            input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

            # 注意：这里需要真实的标签来计算准确率
            total += 1

    return model

# --- Training and Prediction ---
dataset_obj = TextDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset_obj, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

# 模型比较
models_config = {
    'RNN': RNNClassifier,
    'LSTM': LSTMClassifier,
    'GRU': GRUClassifier
}

results = {}

for name, ModelClass in models_config.items():
    print(f"\n===训练 {name} 模型===")

    model = ModelClass(vocab_size, embedding_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_time = train_model(model, dataloader, criterion, optimizer)

    results[name] = {
        'model': model,
        'training_time': training_time
    }

    print(f"{name} 训练完成，耗时：{training_time:.2f}秒")

# 预测函数
def predict_with_model(model, text, char_to_index, max_len, index_to_label):
    model.eval()
    with torch.no_grad():
        indices = [char_to_index.get(char, 0) for char in text[:max_len]]
        indices += [0] * (max_len - len(indices))
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

        predicted_label = index_to_label[predicted.item()]
        return predicted_label

# 测试预测
test_texts = ["帮我导航到北京", "查询明天北京的天气", "把空调开到28度", "今天星期几"]
index_to_label = {i: label for label, i in label_to_index.items()}

print("\n=== 预测结果对比 ===")
for text in test_texts:
    print(f"\n输入文本：'{text}'")
    for name, result in results.items():
        prediction = predict_with_model(result['model'], text, char_to_index, max_len, index_to_label)
        print(f"{name}: {prediction}")
