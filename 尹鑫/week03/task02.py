import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ======================
# 1. 数据读取
# ======================
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
index_to_label = {i: label for label, i in label_to_index.items()}
numerical_labels = [label_to_index[label] for label in string_labels]

# ======================
# 2. 构建字符词表
# ======================
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40

# ======================
# 3. Dataset
# ======================
class CharDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [char_to_index.get(c, 0) for c in text[:max_len]]
        indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

dataset = CharDataset(texts, numerical_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ======================
# 4. 三种模型
# ======================
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.rnn(emb)
        return self.fc(h.squeeze(0))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        return self.fc(h.squeeze(0))


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        return self.fc(h.squeeze(0))

# ======================
# 5. 训练 + 评估函数
# ======================
def train_and_evaluate(ModelClass, model_name):
    model = ModelClass(vocab_size, 64, 128, len(label_to_index))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(4):
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 评估（accuracy）
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"{model_name} Accuracy: {acc:.4f}")
    return acc

# ======================
# 6. 对照实验
# ======================
results = {}
results["RNN"] = train_and_evaluate(RNNClassifier, "RNN")
results["LSTM"] = train_and_evaluate(LSTMClassifier, "LSTM")
results["GRU"] = train_and_evaluate(GRUClassifier, "GRU")

print("\n===== 最终精度对比 =====")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
