import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ===================== 1. 数据加载与预处理 =====================
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签编码
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}

# 字符编码
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index)
max_len = 40  # 文本最大长度

# ===================== 2. 自定义数据集 =====================
class CharRNNDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 截断+补零
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# 初始化数据集和DataLoader
dataset = CharRNNDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ===================== 3. 定义4个模型类 =====================
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # 基础RNN
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden_state = self.rnn(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)  # GRU
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# ===================== 4. 训练与评估函数 =====================
def train_model(model, dataloader, criterion, optimizer, num_epochs=4):
    """通用训练函数"""
    model.train()
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if idx % 50 == 0:
                print(f"Batch {idx}, Loss: {loss.item():.4f}")
        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return loss_history

def evaluate_model(model, dataloader):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# ===================== 5. 初始化模型并训练 =====================
# 超参数
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
lr = 0.001
num_epochs = 4

# 存储实验结果
results = {}

# --- 训练基础RNN ---
print("\n===== 训练基础RNN =====")
rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
rnn_criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=lr)
rnn_loss = train_model(rnn_model, dataloader, rnn_criterion, rnn_optimizer, num_epochs)
rnn_acc = evaluate_model(rnn_model, dataloader)
results["RNN"] = {"loss": rnn_loss, "accuracy": rnn_acc}

# --- 训练LSTM ---
print("\n===== 训练LSTM =====")
lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
lstm_criterion = nn.CrossEntropyLoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=lr)
lstm_loss = train_model(lstm_model, dataloader, lstm_criterion, lstm_optimizer, num_epochs)
lstm_acc = evaluate_model(lstm_model, dataloader)
results["LSTM"] = {"loss": lstm_loss, "accuracy": lstm_acc}

# --- 训练GRU ---
print("\n===== 训练GRU =====")
gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
gru_criterion = nn.CrossEntropyLoss()
gru_optimizer = optim.Adam(gru_model.parameters(), lr=lr)
gru_loss = train_model(gru_model, dataloader, gru_criterion, gru_optimizer, num_epochs)
gru_acc = evaluate_model(gru_model, dataloader)
results["GRU"] = {"loss": gru_loss, "accuracy": gru_acc}

# ===================== 6. 实验结果对比 =====================
print("\n================ 实验结果汇总 ================")
for model_name, metrics in results.items():
    avg_loss = np.mean(metrics["loss"])
    acc = metrics["accuracy"]
    print(f"{model_name} - 平均损失: {avg_loss:.4f}, 准确率: {acc:.2f}%")

# ===================== 7. 预测函数（复用） =====================
def classify_text(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)
    return index_to_label[predicted_idx.item()]

# 测试示例
new_text = "帮我导航到北京"
print(f"\n===== 预测示例 =====")
print(f"输入: {new_text}")
print(f"RNN预测: {classify_text(new_text, rnn_model, char_to_index, max_len, index_to_label)}")
print(f"LSTM预测: {classify_text(new_text, lstm_model, char_to_index, max_len, index_to_label)}")
print(f"GRU预测: {classify_text(new_text, gru_model, char_to_index, max_len, index_to_label)}")
