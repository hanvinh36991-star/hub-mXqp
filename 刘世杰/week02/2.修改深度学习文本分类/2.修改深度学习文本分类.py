import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# 解决matplotlib绘图报错问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#图表无法显示中文解决方法（ Matplotlib显示是DejaVu Sans 字体）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 数据加载与预处理
dataset = pd.read_csv("dataset.csv", sep=",", header=None, nrows=100, encoding='gbk')
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签映射
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 字符映射
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# 定义可配置层数的分类器
class ConfigurableClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        可配置层数的分类器
        :param input_dim: 输入维度（词汇表大小）
        :param hidden_dims: 隐藏层维度列表，如[128]（单层）、[256, 128]（两层）、[512, 256, 128]（三层）
        :param output_dim: 输出维度（类别数）
        """
        super(ConfigurableClassifier, self).__init__()
        layers = []
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        # 隐藏层之间的连接
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())
        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # 组合所有层
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 训练函数（返回训练过程的Loss列表）
def train_model(hidden_dims, dataloader, vocab_size, output_dim, num_epochs=10, lr=0.01):
    """
    训练模型并返回Loss记录
    """
    # 创建模型
    model = ConfigurableClassifier(vocab_size, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 记录每个epoch的Loss
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 计算当前epoch的平均Loss
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"模型配置 {hidden_dims} | Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return epoch_losses, model


# 初始化数据集和数据加载器
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)
output_dim = len(label_to_index)

# 定义不同的模型配置（对比实验）
model_configs = {
    "单层-64节点": [64],  # 基础配置1
    "单层-128节点": [128],  # 基础配置2（原代码）
    "单层-256节点": [256],  # 基础配置3
    "两层-256-128节点": [256, 128],  # 两层配置
    "三层-512-256-128节点": [512, 256, 128]  # 三层配置
}

# 训练所有配置并记录Loss
loss_records = {}
trained_models = {}
num_epochs = 10
for config_name, hidden_dims in model_configs.items():
    print(f"\n========== 开始训练 {config_name} ==========")
    loss_records[config_name], trained_models[config_name] = train_model(
        hidden_dims, dataloader, vocab_size, output_dim, num_epochs=num_epochs
    )

# 可视化不同配置的Loss变化
plt.figure(figsize=(12, 6))
for config_name, losses in loss_records.items():
    plt.plot(range(1, num_epochs + 1), losses, label=config_name, marker='o')

plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("不同模型配置的Loss变化对比")
plt.legend()
plt.grid(True)
plt.show()


# 分类函数（保持不变）
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 测试最优模型（可选，这里选Loss最低的模型）
index_to_label = {i: label for label, i in label_to_index.items()}
best_config = min(loss_records.keys(), key=lambda x: loss_records[x][-1])
best_model = trained_models[best_config]
print(f"\n========== 使用最优模型（{best_config}）测试 ==========")

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
