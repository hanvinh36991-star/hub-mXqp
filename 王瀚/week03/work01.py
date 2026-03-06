import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import time

# 设置随机种子确保可复现
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 数据预处理（复用你的逻辑）
# ==========================================
try:
    dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
except:
    # 如果没有文件，创建示例数据（仅用于演示代码结构）
    print("注意：未找到数据文件，使用示例数据...")
    data = {
        0: ["查询天气", "导航到北京", "播放音乐", "设置闹钟", "发送短信"] * 100,
        1: ["weather", "navigation", "music", "alarm", "message"] * 100
    }
    dataset = pd.DataFrame(data)

texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
index_to_label = {i: label for label, i in label_to_index.items()}
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建字符词典
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 20
num_classes = len(label_to_index)


# ==========================================
# 2. 数据集类（复用你的实现）
# ==========================================
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


# 划分数据集 8:1:1
full_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ==========================================
# 3. 三个模型定义（RNN vs LSTM vs GRU）
# ==========================================
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, nonlinearity='tanh')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        output, hidden = self.rnn(embedded)  # hidden: [num_layers, batch, hidden]
        # 取最后一个时间步的隐藏状态（或最后一个layer的hidden）
        final_hidden = hidden[-1]  # [batch, hidden_dim]
        final_hidden = self.dropout(final_hidden)
        return self.fc(final_hidden)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        # LSTM 返回 hidden: [num_layers, batch, hidden], 取最后一层
        final_hidden = hidden[-1]
        final_hidden = self.dropout(final_hidden)
        return self.fc(final_hidden)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        # GRU 返回 hidden: [num_layers, batch, hidden], 取最后一层
        final_hidden = hidden[-1]
        final_hidden = self.dropout(final_hidden)
        return self.fc(final_hidden)


# ==========================================
# 4. 训练与评估函数
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device='cpu'):
    model.to(device)
    history = {
        'train_loss': [],
        'val_acc': [],
        'train_acc': []
    }

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")

    return history


def evaluate_model(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# ==========================================
# 5. 超参数设置与模型初始化
# ==========================================
embedding_dim = 64
hidden_dim = 128
learning_rate = 0.001
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {device}")
print(f"词汇表大小: {vocab_size}")
print(f"类别数: {num_classes}")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 初始化三个模型
models = {
    'RNN': RNNClassifier(vocab_size, embedding_dim, hidden_dim, num_classes).to(device),
    'LSTM': LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_classes).to(device),
    'GRU': GRUClassifier(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)
}

histories = {}
test_results = {}
training_times = {}

criterion = nn.CrossEntropyLoss()

# ==========================================
# 6. 训练三个模型
# ==========================================
for name, model in models.items():
    print(f"\n{'=' * 50}")
    print(f"开始训练 {name} 模型")
    print('=' * 50)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)
    end_time = time.time()

    # 测试集评估
    test_acc = evaluate_model(model, test_loader, device)

    histories[name] = history
    test_results[name] = test_acc
    training_times[name] = end_time - start_time

    print(f"\n{name} 最终结果:")
    print(f"测试集准确率: {test_acc:.2f}%")
    print(f"训练时间: {training_times[name]:.2f}秒")

# ==========================================
# 7. 可视化对比
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示为方块

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 图1: 测试准确率对比（柱状图）
ax1 = axes[0, 0]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax1.bar(test_results.keys(), test_results.values(), color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('准确率 (%)', fontsize=12)
ax1.set_title('模型测试准确率对比', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 100)
for i, (name, acc) in enumerate(test_results.items()):
    ax1.text(i, acc + 2, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 图2: 验证准确率训练曲线（折线图）
ax2 = axes[0, 1]
for name, history in histories.items():
    ax2.plot(range(1, epochs + 1), history['val_acc'], marker='o', linewidth=2, label=name)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('验证准确率 (%)', fontsize=12)
ax2.set_title('验证准确率收敛过程', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3: 训练损失下降曲线
ax3 = axes[1, 0]
for name, history in histories.items():
    ax3.plot(range(1, epochs + 1), history['train_loss'], marker='s', linewidth=2, label=name)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('训练损失', fontsize=12)
ax3.set_title('训练损失变化趋势', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4: 训练时间对比
ax4 = axes[1, 1]
bars = ax4.bar(training_times.keys(), training_times.values(), color=colors, alpha=0.8, edgecolor='black')
ax4.set_ylabel('时间 (秒)', fontsize=12)
ax4.set_title('训练耗时对比', fontsize=14, fontweight='bold')
for i, (name, t) in enumerate(training_times.items()):
    ax4.text(i, t + 0.5, f'{t:.1f}s', ha='center', va='bottom', fontsize=10)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 8. 最终对比表格
# ==========================================
print("\n" + "=" * 60)
print("最终结果对比表")
print("=" * 60)
print(f"{'模型':<10}{'测试准确率':<15}{'训练时间(s)':<15}{'最好验证准确率':<15}")
print("-" * 60)
for name in ['RNN', 'LSTM', 'GRU']:
    best_val = max(histories[name]['val_acc'])
    print(f"{name:<10}{test_results[name]:<15.2f}{training_times[name]:<15.2f}{best_val:<15.2f}")
print("=" * 60)

# ==========================================
# 9. 单条预测示例（使用最优模型）
# ==========================================
best_model_name = max(test_results, key=test_results.get)
print(f"\n最优模型: {best_model_name} (准确率: {test_results[best_model_name]:.2f}%)")

best_model = models[best_model_name]
best_model.eval()


def predict_text(text, model, char_to_index, max_len):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    return index_to_label[predicted.item()]


# 测试样例
test_samples = ["帮我导航到北京", "查询明天北京的天气", "播放周杰伦的歌"]
print("\n预测示例:")
for text in test_samples:
    pred = predict_text(text, best_model, char_to_index, max_len)
    print(f"输入: '{text}' -> 预测: '{pred}'")
