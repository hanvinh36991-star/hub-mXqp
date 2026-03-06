import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
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

class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


full_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(full_dataset, batch_size=16, shuffle=True)  # Batch size调小点适应小数据

class TextClassifier(nn.Module):
    def __init__(self, model_type, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.model_type = model_type

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 根据 model_type 初始化不同的循环层
        if model_type == 'RNN':
            self.rnn_layer = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == 'GRU':
            self.rnn_layer = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == 'LSTM':
            self.rnn_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("model_type must be RNN, GRU, or LSTM")

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(x)

        # 运行循环层
        if self.model_type == 'LSTM':
            # LSTM 返回 (output, (hidden, cell))
            _, (hidden, _) = self.rnn_layer(embedded)
        else:
            # RNN 和 GRU 返回 (output, hidden)
            _, hidden = self.rnn_layer(embedded)

        # hidden shape: [num_layers * num_directions, batch, hidden_dim]
        # 只取最后一层的 hidden state
        # squeeze(0) 移除第一维 -> [batch, hidden_dim]
        out = self.fc(hidden.squeeze(0))
        return out
      
def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train_model(model_type, epochs=10):
    print(f"\n--- 开始训练模型: {model_type} ---")

    # 超参数
    embedding_dim = 64
    hidden_dim = 128
    output_dim = len(label_to_index)

    model = TextClassifier(model_type, vocab_size, embedding_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    end_time = time.time()
    acc = evaluate_accuracy(model, dataloader)

    print(
        f"模型: {model_type} | 最终 Loss: {running_loss / len(dataloader):.4f} | 准确率: {acc * 100:.2f}% | 耗时: {end_time - start_time:.2f}s")
    return model, acc


# 统一训练轮数
NUM_EPOCHS = 20  
# 分别训练三个模型
rnn_model, rnn_acc = train_model('RNN', epochs=NUM_EPOCHS)
gru_model, gru_acc = train_model('GRU', epochs=NUM_EPOCHS)
lstm_model, lstm_acc = train_model('LSTM', epochs=NUM_EPOCHS)

print("\n" + "=" * 30)
print("实验结果汇总")
print("=" * 30)
print(f"{'模型类型':<10} | {'准确率':<10}")
print("-" * 25)
print(f"{'RNN':<10} | {rnn_acc * 100:.2f}%")
print(f"{'GRU':<10} | {gru_acc * 100:.2f}%")
print(f"{'LSTM':<10} | {lstm_acc * 100:.2f}%")
print("-" * 25)

best_model = lstm_model
if gru_acc > lstm_acc and gru_acc > rnn_acc:
    best_model = gru_model
elif rnn_acc > lstm_acc and rnn_acc > gru_acc:
    best_model = rnn_model

def predict(text, model):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return index_to_label[predicted.item()]

test_text = "帮我导航到天安门"
print(f"\n测试最佳模型 ({best_model.model_type}) -> 输入: '{test_text}' -> 预测: {predict(test_text, best_model)}")
