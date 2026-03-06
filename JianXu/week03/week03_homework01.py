import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("./Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0, '<unk>': 1}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40

class CharLSTMDataset(Dataset):
    '''
    数据格式： input (B, max_len) , lables： (B, )
    '''
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, self.char_to_index['<unk>']) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# --- NEW LSTM Model Class ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x) # (B, L) ->（B, L, E）
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)  # (B, L, E) -> (B, L, H), (1, B, H)
        out = self.fc(hidden_state.squeeze(0)) # (B, C)
        return out

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # (B, L, D)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # (B, L) -> (B, L, D)
        output, h_n  = self.rnn(embedded) # (B, L, D) -> (B, L, H), (1, B, H)
        out = self.fc(h_n.squeeze(0))    # (1, B, H) -> (B, H)
        return out

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # (B, L, D)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)  
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # (B, L) -> (B, L, D)
        output, h_n  = self.gru(embedded) # (B, L, D) -> (B, L, H), (1, B, H)
        out = self.fc(h_n[-1])    # (1, B, H) -> (B, H)
        return out


# --- Training and Prediction ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

# model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 8
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
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
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

'''
GRU:
Epoch [8/8], Loss: 0.0576
输入 '帮我导航到北京' 预测为: 'Travel-Query'
输入 '查询明天北京的天气' 预测为: 'Weather-Query'
LSTM:
Epoch [8/8], Loss: 0.1797
输入 '帮我导航到北京' 预测为: 'Travel-Query'
输入 '查询明天北京的天气' 预测为: 'Weather-Query'
RNN:
Epoch [8/8], Loss: 2.3567
输入 '帮我导航到北京' 预测为: 'FilmTele-Play'
输入 '查询明天北京的天气' 预测为: 'FilmTele-Play'
'''