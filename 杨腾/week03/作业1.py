import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. 固定随机种子（保证可复现）
torch.manual_seed(42)


# 2. 数据读取与预处理
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40


# 3. 数据集定义
class CharSeqDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(c, 0) for c in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

dataset = CharSeqDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 4. 模型定义
# ---- RNN ----
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        return self.fc(hidden.squeeze(0))

# ---- LSTM ----
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden.squeeze(0))

# ---- GRU ----
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.gru(x)
        return self.fc(hidden.squeeze(0))


# 5. 训练与评估函数
def train_and_eval(model, dataloader, model_name, epochs=4):
    print(f"\n{'='*20} {model_name} {'='*20}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if idx % 50 == 0:
                print(f"Batch {idx}, Loss: {loss.item():.4f}")

        acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {total_loss/len(dataloader):.4f}, "
              f"Accuracy: {acc:.4f}")

# 6. 依次训练三种模型
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

train_and_eval(rnn_model, dataloader, "RNN")
train_and_eval(lstm_model, dataloader, "LSTM")
train_and_eval(gru_model, dataloader, "GRU")


# 7. 定义通用预测函数
def classify_text(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(c, 0) for c in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    x = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(x)

    _, pred = torch.max(output, 1)
    return index_to_label[pred.item()]

# 8.输出预测结果对比
test_texts = [
    "帮我导航到北京",
    "查询明天北京的天气"
]
print("\n========== 三种模型预测对比 ==========")
for text in test_texts:
    print(f"\n输入文本：{text}")
    print(f"RNN  预测结果：{classify_text(text, rnn_model, char_to_index, max_len, index_to_label)}")
    print(f"LSTM 预测结果：{classify_text(text, lstm_model, char_to_index, max_len, index_to_label)}")
    print(f"GRU  预测结果：{classify_text(text, gru_model, char_to_index, max_len, index_to_label)}")
