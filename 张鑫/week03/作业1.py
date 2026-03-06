import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

# ==========================================
# 1. 数据准备与预处理
# ==========================================

# 读取数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 建立标签映射
label_to_index = {label: i for i, label in enumerate(sorted(set(string_labels)))}
index_to_label = {i: label for label, i in label_to_index.items()}
numerical_labels = [label_to_index[label] for label in string_labels]

# 建立字符词表
char_to_index = {'<pad>': 0, '<unk>': 1}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 15
output_dim = len(label_to_index)


# 自定义数据集类
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
        # 字符转索引，超长截断，不足补0
        indices = [self.char_to_index.get(char, 1) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# ==========================================
# 2. 通用分类模型类 (支持 RNN / LSTM / GRU)
# ==========================================

class TextClassifier(nn.Module):
    def __init__(self, model_type, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if model_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)

        # 不同模型的返回结构处理
        if self.model_type == 'LSTM':
            # LSTM 返回 (output, (hn, cn))
            _, (hn, cn) = self.rnn(embedded)
        else:
            # RNN 和 GRU 返回 (output, hn)
            _, hn = self.rnn(embedded)

        # 取最后一层隐藏状态进行分类 [batch_size, hidden_dim]
        # hn 的维度是 [num_layers, batch, hidden_dim]，取 squeeze(0)
        out = self.fc(hn.squeeze(0))
        return out


# ==========================================
# 3. 训练与对比实验函数
# ==========================================

def train_model(model_type, train_loader):
    print(f"\n--- 正在启动 {model_type} 模型训练 ---")
    model = TextClassifier(model_type, vocab_size, 64, 128, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 30
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.4f}")

    end_time = time.time()
    print(f"{model_type} 训练完成，总耗时: {end_time - start_time:.2f}秒")
    return model


# ==========================================
# 4. 执行对比实验
# ==========================================

# 创建数据加载器
full_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

# 训练三个模型并保存
trained_models = {}
for m_type in ['RNN', 'LSTM', 'GRU']:
    trained_models[m_type] = train_model(m_type, train_loader)

# ==========================================
# 5. 预测 10 条数据进行对比
# ==========================================

test_texts = [
    "帮我定个明天早上八点的闹钟",  # 预期: Alarm-Update
    "我想听周杰伦的青花瓷",  # 预期: Music-Play
    "今天北京的天气怎么样",  # 预期: Weather-Query
    "从这里去机场怎么走",  # 预期: Travel-Query
    "把客厅的空调调到26度",  # 预期: HomeAppliance-Control
    "给我播放最新的新闻节目",  # 预期: Video-Play 或 Radio-Listen
    "农历五月初五是什么节日",  # 预期: Calendar-Query
    "我要看周星驰的喜剧电影",  # 预期: FilmTele-Play
    "帮我打开加湿器",  # 预期: HomeAppliance-Control
    "查询北京飞上海的航班情况"  # 预期: Travel-Query
]


def predict(text, model):
    model.eval()
    with torch.no_grad():
        indices = [char_to_index.get(char, 1) for char in text[:max_len]]
        indices += [0] * (max_len - len(indices))
        input_tensor = torch.tensor(indices).unsqueeze(0)
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return index_to_label[predicted.item()]


print("\n" + "=" * 80)
print(f"{'待预测文本':<25} | {'RNN预测结果':<15} | {'LSTM预测结果':<15} | {'GRU预测结果':<15}")
print("-" * 80)

for t in test_texts:
    res_rnn = predict(t, trained_models['RNN'])
    res_lstm = predict(t, trained_models['LSTM'])
    res_gru = predict(t, trained_models['GRU'])
    print(f"{t:<25} | {res_rnn:<15} | {res_lstm:<15} | {res_gru:<15}")

print("=" * 80)
