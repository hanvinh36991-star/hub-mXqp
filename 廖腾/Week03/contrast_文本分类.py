import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import time

# 数据加载及预处理
dataset = pd.read_csv("E:\YunZjDownload\pytorch-learning\Ai_study\dataset.csv", sep="\t", header=None)
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

max_len = 40


# 定义输入模型的数据格式
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


# --- RNN Model Class ---
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 num_layers=1, dropout=0.0, bidirectional=False):
        super(RNNClassifier, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # RNN层
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            nonlinearity='tanh'  # 可以是'tanh'或'relu'
        )

        # 全连接层
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_output_dim, output_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 嵌入层
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # RNN层
        rnn_output, hidden = self.rnn(embedded)

        # 处理隐藏状态
        if self.bidirectional:
            # 双向：连接最后两个方向的隐藏状态
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            # 单向：取最后一个层的隐藏状态
            hidden = hidden[-1]

        # 全连接层
        output = self.fc(hidden)

        return output


# --- GRU Model Class ---
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 num_layers=1, dropout=0.0, bidirectional=False):
        super(GRUClassifier, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # GRU层
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 全连接层
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(gru_output_dim, output_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 嵌入层
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # GRU层
        gru_output, hidden = self.gru(embedded)

        # 处理隐藏状态
        if self.bidirectional:
            # 双向：连接最后两个方向的隐藏状态
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            # 单向：取最后一个层的隐藏状态
            hidden = hidden[-1]

        # 全连接层
        output = self.fc(hidden)

        return output


# --- LSTM Model Class ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 num_layers=1, dropout=0.0, bidirectional=False):
        super(LSTMClassifier, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 全连接层
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, output_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 嵌入层
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # LSTM层
        lstm_output, (hidden, cell) = self.lstm(embedded)

        # 处理隐藏状态
        if self.bidirectional:
            # 双向：连接最后两个方向的隐藏状态
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            # 单向：取最后一个层的隐藏状态
            hidden = hidden[-1]

        # 全连接层
        output = self.fc(hidden)

        return output


# --- 参数计算函数 ---
def calculate_parameters(model):
    """详细计算模型各部分参数"""
    total_params = 0
    param_details = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            param_details[name] = num_params

    return total_params, param_details


# --- 数据分割（训练集和验证集）---
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, numerical_labels, test_size=0.2, random_state=42, stratify=numerical_labels
)

# --- 创建模型数据集和DataLoader ---
train_dataset = CharDataset(train_texts, train_labels, char_to_index, max_len)
val_dataset = CharDataset(val_texts, val_labels, char_to_index, max_len)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- 超参数设置（完全相同配置）---
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
num_layers = 2  # 都使用2层
dropout_rate = 0.3  # 相同的dropout率
bidirectional = False  # 都使用单向
criterion = nn.CrossEntropyLoss()
num_epochs = 20
learning_rate = 0.001

# 创建三种相同配置的模型
model_RNN = RNNClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_layers=num_layers,
    dropout=dropout_rate,
    bidirectional=bidirectional
)

model_GRU = GRUClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_layers=num_layers,
    dropout=dropout_rate,
    bidirectional=bidirectional
)

model_LSTM = LSTMClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_layers=num_layers,
    dropout=dropout_rate,
    bidirectional=bidirectional
)

# 使用相同的优化器配置
optimizer_RNN = optim.Adam(model_RNN.parameters(), lr=learning_rate)
optimizer_GRU = optim.Adam(model_GRU.parameters(), lr=learning_rate)
optimizer_LSTM = optim.Adam(model_LSTM.parameters(), lr=learning_rate)

# --- 详细参数对比 ---
print("=" * 70)
print("RNN、GRU和LSTM三种模型对比（相同配置）")
print("=" * 70)
print(f"配置参数:")
print(f"  - 词汇表大小: {vocab_size}")
print(f"  - 嵌入维度: {embedding_dim}")
print(f"  - 隐藏层维度: {hidden_dim}")
print(f"  - 输出维度: {output_dim}")
print(f"  - 层数: {num_layers}")
print(f"  - Dropout率: {dropout_rate}")
print(f"  - 双向: {bidirectional}")
print()

# 计算参数
rnn_total, rnn_details = calculate_parameters(model_RNN)
gru_total, gru_details = calculate_parameters(model_GRU)
lstm_total, lstm_details = calculate_parameters(model_LSTM)

print("模型参数详情:")
print("-" * 60)

print("RNN参数分布:")
for name, num in rnn_details.items():
    print(f"  {name}: {num:>8,}")

print("\nGRU参数分布:")
for name, num in gru_details.items():
    print(f"  {name}: {num:>8,}")

print("\nLSTM参数分布:")
for name, num in lstm_details.items():
    print(f"  {name}: {num:>8,}")

print("\n" + "=" * 70)
print("参数统计对比")
print("=" * 70)
print(f"{'模型':<10} {'总参数':<12} {'RNN层参数':<15} {'相对RNN减少':<15}")
print("-" * 70)


# 计算RNN层参数（去掉嵌入层和全连接层）
def get_rnn_layer_params(details):
    rnn_params = 0
    for name in details:
        if 'rnn' in name or 'gru' in name or 'lstm' in name:
            rnn_params += details[name]
    return rnn_params


rnn_layer_params = get_rnn_layer_params(rnn_details)
gru_layer_params = get_rnn_layer_params(gru_details)
lstm_layer_params = get_rnn_layer_params(lstm_details)

print(f"{'RNN':<10} {rnn_total:<12,} {rnn_layer_params:<15,} {'-':<15}")
print(
    f"{'GRU':<10} {gru_total:<12,} {gru_layer_params:<15,} {f'{(rnn_layer_params - gru_layer_params) / rnn_layer_params * 100:.1f}%':<15}")
print(
    f"{'LSTM':<10} {lstm_total:<12,} {lstm_layer_params:<15,} {f'{(rnn_layer_params - lstm_layer_params) / rnn_layer_params * 100:.1f}%':<15}")

# 理论参数计算
print("\n理论参数分析:")
print("-" * 40)
embedding_params = vocab_size * embedding_dim
fc_params = (hidden_dim * output_dim + output_dim)

# 单层参数计算
print(f"单层参数计算公式:")
print(f"  RNN: (input_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim)")
print(f"  GRU: 3 × (input_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim)")
print(f"  LSTM: 4 × (input_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim)")


# --- 定义模型训练和评估函数 ---
def train_and_evaluate(model, train_dataloader, val_dataloader, num_epochs, criterion, optimizer, model_name):
    train_loss_history = []
    val_accuracy_history = []
    train_acc_history = []

    print(f"\n--- 训练{model_name}模型 ---")
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for idx, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)

            optimizer.step()
            running_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_dataloader)
        train_accuracy = train_correct / train_total
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_accuracy)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        val_accuracy_history.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'best_{model_name}_model.pth')
            print(f"  ✓ 保存最佳模型，验证准确率: {val_accuracy:.4f}")

    print(f"\n{model_name}最佳验证准确率: {best_val_accuracy:.4f}")
    return train_loss_history, train_acc_history, val_accuracy_history, best_val_accuracy


# --- 训练三个模型 ---
print("\n" + "=" * 70)
print("开始训练三种模型...")
print("=" * 70)

train_loss_RNN, train_acc_RNN, val_acc_RNN, best_RNN = train_and_evaluate(
    model_RNN, train_dataloader, val_dataloader,
    num_epochs, criterion, optimizer_RNN, "RNN"
)

train_loss_GRU, train_acc_GRU, val_acc_GRU, best_GRU = train_and_evaluate(
    model_GRU, train_dataloader, val_dataloader,
    num_epochs, criterion, optimizer_GRU, "GRU"
)

train_loss_LSTM, train_acc_LSTM, val_acc_LSTM, best_LSTM = train_and_evaluate(
    model_LSTM, train_dataloader, val_dataloader,
    num_epochs, criterion, optimizer_LSTM, "LSTM"
)

# --- 绘制对比图 ---
plt.figure(figsize=(18, 5))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 损失曲线
plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs + 1), train_loss_RNN, 'g-', label=f'RNN Loss', linewidth=2)
plt.plot(range(1, num_epochs + 1), train_loss_GRU, 'b-', label=f'GRU Loss', linewidth=2)
plt.plot(range(1, num_epochs + 1), train_loss_LSTM, 'r-', label=f'LSTM Loss', linewidth=2)
plt.title('Training Loss Comparison', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 训练准确率
plt.subplot(1, 3, 2)
plt.plot(range(1, num_epochs + 1), train_acc_RNN, 'g--', label=f'RNN Train Acc', linewidth=2)
plt.plot(range(1, num_epochs + 1), train_acc_GRU, 'b--', label=f'GRU Train Acc', linewidth=2)
plt.plot(range(1, num_epochs + 1), train_acc_LSTM, 'r--', label=f'LSTM Train Acc', linewidth=2)
plt.title('Training Accuracy Comparison', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 验证准确率
plt.subplot(1, 3, 3)
plt.plot(range(1, num_epochs + 1), val_acc_RNN, 'g-', label=f'RNN Val Acc', linewidth=2, alpha=0.8)
plt.plot(range(1, num_epochs + 1), val_acc_GRU, 'b-', label=f'GRU Val Acc', linewidth=2, alpha=0.8)
plt.plot(range(1, num_epochs + 1), val_acc_LSTM, 'r-', label=f'LSTM Val Acc', linewidth=2, alpha=0.8)
plt.title('Validation Accuracy Comparison', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# --- 推理时间测量 ---
def estimate_inference_time(model, dataloader, num_runs=10):
    model.eval()
    total_time = 0

    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            for inputs, labels in dataloader:
                _ = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)

    return total_time / num_runs


print("\n" + "=" * 70)
print("推理时间测量...")
print("=" * 70)

rnn_inference_time = estimate_inference_time(model_RNN, val_dataloader)
gru_inference_time = estimate_inference_time(model_GRU, val_dataloader)
lstm_inference_time = estimate_inference_time(model_LSTM, val_dataloader)

# --- 性能总结 ---
print("\n" + "=" * 70)
print("性能总结对比")
print("=" * 70)

print(f"{'指标':<15} {'RNN':<10} {'GRU':<10} {'LSTM':<10}")
print("-" * 70)

# 计算最终性能
rnn_final_train_acc = train_acc_RNN[-1]
gru_final_train_acc = train_acc_GRU[-1]
lstm_final_train_acc = train_acc_LSTM[-1]

rnn_final_val_acc = val_acc_RNN[-1]
gru_final_val_acc = val_acc_GRU[-1]
lstm_final_val_acc = val_acc_LSTM[-1]

print(f"{'参数量':<15} {rnn_total:<10,} {gru_total:<10,} {lstm_total:<10,}")
print(f"{'训练准确率':<15} {rnn_final_train_acc:.4f} {gru_final_train_acc:.4f} {lstm_final_train_acc:.4f}")
print(f"{'验证准确率':<15} {rnn_final_val_acc:.4f} {gru_final_val_acc:.4f} {lstm_final_val_acc:.4f}")
print(f"{'最佳验证准确率':<15} {best_RNN:.4f} {best_GRU:.4f} {best_LSTM:.4f}")
print(f"{'推理时间(s)':<15} {rnn_inference_time:.4f} {gru_inference_time:.4f} {lstm_inference_time:.4f}")

print("\n" + "=" * 70)
print("相对性能对比（以RNN为基准）")
print("=" * 70)
print(f"{'指标':<15} {'GRU vs RNN':<15} {'LSTM vs RNN':<15}")
print("-" * 70)

# 相对于RNN的改进
gru_acc_improvement = (best_GRU - best_RNN) * 100
lstm_acc_improvement = (best_LSTM - best_RNN) * 100

gru_speedup = (rnn_inference_time - gru_inference_time) / rnn_inference_time * 100
lstm_speedup = (rnn_inference_time - lstm_inference_time) / rnn_inference_time * 100

gru_param_change = (gru_total - rnn_total) / rnn_total * 100
lstm_param_change = (lstm_total - rnn_total) / rnn_total * 100

print(f"{'准确率提升':<15} {gru_acc_improvement:+.2f}%{' ':>5} {lstm_acc_improvement:+.2f}%{' ':>5}")
print(f"{'速度提升':<15} {gru_speedup:+.2f}%{' ':>5} {lstm_speedup:+.2f}%{' ':>5}")
print(f"{'参数增加':<15} {gru_param_change:+.2f}%{' ':>5} {lstm_param_change:+.2f}%{' ':>5}")

# --- 预测部分 ---
index_to_label = {i: label for label, i in label_to_index.items()}


def classify_text(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)

    return index_to_label[predicted_index.item()], confidence.item()


# 使用训练好的模型进行预测
test_texts = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "播放一首周杰伦的歌",
    "打开微信",
    "今天天气怎么样",
    "设置明天早上7点的闹钟",
    "给妈妈打个电话",
    "翻译你好为英语"
]

print("\n" + "=" * 70)
print("预测测试对比")
print("=" * 70)

# 收集所有模型的预测结果
results = []

for text in test_texts:
    if len(text) > 0:
        rnn_pred, rnn_conf = classify_text(text, model_RNN, char_to_index, max_len, index_to_label)
        gru_pred, gru_conf = classify_text(text, model_GRU, char_to_index, max_len, index_to_label)
        lstm_pred, lstm_conf = classify_text(text, model_LSTM, char_to_index, max_len, index_to_label)

        results.append({
            'text': text,
            'rnn': (rnn_pred, rnn_conf),
            'gru': (gru_pred, gru_conf),
            'lstm': (lstm_pred, lstm_conf)
        })

# 显示预测结果对比
print(f"{'文本':<25} {'RNN预测':<15} {'置信度':<8} {'GRU预测':<15} {'置信度':<8} {'LSTM预测':<15} {'置信度':<8}")
print("-" * 90)

for result in results:
    text = result['text']
    rnn_pred, rnn_conf = result['rnn']
    gru_pred, gru_conf = result['gru']
    lstm_pred, lstm_conf = result['lstm']

    print(f"{text:<25} {rnn_pred:<15} {rnn_conf:.3f}   {gru_pred:<15} {gru_conf:.3f}   {lstm_pred:<15} {lstm_conf:.3f}")

# 计算一致性
print("\n" + "=" * 70)
print("预测一致性分析")
print("=" * 70)

consistent_predictions = 0
for result in results:
    predictions = [result['rnn'][0], result['gru'][0], result['lstm'][0]]
    if len(set(predictions)) == 1:
        consistent_predictions += 1

print(f"总测试样本: {len(results)}")
print(f"三模型一致预测: {consistent_predictions}")
print(f"一致性比例: {consistent_predictions / len(results) * 100:.1f}%")

# 创建性能对比柱状图
plt.figure(figsize=(12, 8))

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 参数量对比
axes[0, 0].bar(['RNN', 'GRU', 'LSTM'], [rnn_total, gru_total, lstm_total],
               color=['green', 'blue', 'red'], alpha=0.7)
axes[0, 0].set_title('模型参数量对比', fontsize=14)
axes[0, 0].set_ylabel('参数量', fontsize=12)
for i, v in enumerate([rnn_total, gru_total, lstm_total]):
    axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom')

# 2. 验证准确率对比
axes[0, 1].bar(['RNN', 'GRU', 'LSTM'], [best_RNN, best_GRU, best_LSTM],
               color=['green', 'blue', 'red'], alpha=0.7)
axes[0, 1].set_title('最佳验证准确率对比', fontsize=14)
axes[0, 1].set_ylabel('准确率', fontsize=12)
for i, v in enumerate([best_RNN, best_GRU, best_LSTM]):
    axes[0, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom')

# 3. 推理时间对比
axes[1, 0].bar(['RNN', 'GRU', 'LSTM'], [rnn_inference_time, gru_inference_time, lstm_inference_time],
               color=['green', 'blue', 'red'], alpha=0.7)
axes[1, 0].set_title('推理时间对比', fontsize=14)
axes[1, 0].set_ylabel('时间(秒)', fontsize=12)
for i, v in enumerate([rnn_inference_time, gru_inference_time, lstm_inference_time]):
    axes[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')

# 4. 综合评分（准确率/参数量 * 10000）
scores = [
    best_RNN / rnn_total * 10000,
    best_GRU / gru_total * 10000,
    best_LSTM / lstm_total * 10000
]
axes[1, 1].bar(['RNN', 'GRU', 'LSTM'], scores,
               color=['green', 'blue', 'red'], alpha=0.7)
axes[1, 1].set_title('效率评分（准确率/参数量）', fontsize=14)
axes[1, 1].set_ylabel('评分', fontsize=12)
for i, v in enumerate(scores):
    axes[1, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("总结：")
print("=" * 70)
print("1. RNN: 参数量最少，训练最快，但可能受梯度消失问题影响")
print("2. GRU: 参数量适中，训练速度较快，解决了梯度问题")
print("3. LSTM: 参数量最多，表达能力最强，但训练较慢")
print("\n选择建议：")
print("- 如果追求速度和参数效率：选择RNN或GRU")
print("- 如果追求最高准确率：选择LSTM")
print("- 如果考虑平衡：选择GRU（在准确率和效率之间取得平衡）")