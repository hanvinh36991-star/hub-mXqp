import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 1. 数据加载与预处理（保持原逻辑）
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
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

# 2. 数据集类（保持原逻辑）
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

# 3. 可自定义层数和节点数的分类器
class FlexibleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        :param input_dim: 输入维度（词表大小）
        :param hidden_dims: 隐藏层维度列表，例：[64]（1层64节点）、[128, 64]（2层，128→64）
        :param output_dim: 输出维度（类别数）
        """
        super(FlexibleClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        # 构建输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 4. 封装训练函数（返回每轮Loss）
def train_model(hidden_dims, num_epochs=10, batch_size=32, lr=0.01):
    # 初始化数据集和数据加载器
    char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
    dataloader = DataLoader(char_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型、损失函数、优化器
    output_dim = len(label_to_index)
    model = FlexibleClassifier(vocab_size, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # 记录每轮Loss
    epoch_losses = []
    print(f"\n========== 实验配置：隐藏层 {hidden_dims} ==========")
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
        
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return epoch_losses

# 5. 对比实验配置
experiments = {
    "1层-64节点": [64],
    "1层-128节点": [128],  # 原配置
    "1层-256节点": [256],
    "2层-128→64节点": [128, 64],
    "2层-256→128节点": [256, 128]
}

# 6. 执行所有实验并记录Loss
all_losses = {}
for exp_name, hidden_dims in experiments.items():
    all_losses[exp_name] = train_model(hidden_dims, num_epochs=10)

# 7. 可视化Loss对比
plt.figure(figsize=(10, 6))
for exp_name, losses in all_losses.items():
    plt.plot(range(1, 11), losses, label=exp_name, marker='o')

plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Loss变化对比（不同层数/节点数）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
