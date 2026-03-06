import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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

# max length 最大输入的文本长度
max_len = 40

# 自定义数据集 - 》 为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 统一的写法，底层pytorch 深度学习 / 大模型
class CharLSTMDataset(Dataset):
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

# a = CharLSTMDataset()
# len(a) -> a.__len__
# a[0] -> a.__getitem__


# --- NEW LSTM Model Class ---


# 通用的 RNN 分类器类
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='lstm'):
        super(RNNClassifier, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 根据 rnn_type 选择合适的 RNN 层
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 嵌入层
        embedded = self.embedding(x)
        
        # RNN 层
        if isinstance(self.rnn, nn.LSTM):
            rnn_out, (hidden_state, _) = self.rnn(embedded)
        else:
            rnn_out, hidden_state = self.rnn(embedded)
        
        # 获取最后一个时间步的输出
        out = self.fc(hidden_state.squeeze(0))
        return out

# 数据加载
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

# 训练和预测函数
def train_and_predict(rnn_type):
    model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type=rnn_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 4
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
            if idx % 50 == 0:
                print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    def classify_text(text, model, char_to_index, max_len, index_to_label):
        indices = [char_to_index.get(char, 0) for char in text[:max_len]]
        indices += [0] * (max_len - len(indices))
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        _, predicted_index = torch.max(output, 1)
        predicted_index = predicted_index.item()
        predicted_label = index_to_label[predicted_index]

        return predicted_label

    index_to_label = {i: label for label, i in label_to_index.items()}

    new_text = "帮我导航到北京"
    predicted_class = classify_text(new_text, model, char_to_index, max_len, index_to_label)
    print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

    new_text_2 = "查询明天北京的天气"
    predicted_class_2 = classify_text(new_text_2, model, char_to_index, max_len, index_to_label)
    print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

# 使用 RNN 模型
print("使用 RNN 模型:")
train_and_predict('rnn')

# 使用 GRU 模型
print("使用 GRU 模型:")
train_and_predict('gru')

# 使用 LSTM 模型
print("使用 LSTM 模型:")
train_and_predict('lstm')
