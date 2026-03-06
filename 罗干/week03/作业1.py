import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def read_data(filename):
    dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
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
    return texts, string_labels, index_to_char, max_len

# 构建datas
class NameDataset(Dataset):
    def __init__(self, texts, string_labels, index_to_char, max_len):
        super().__init__()
        self.texts = texts
        self.string_labels = string_labels
        self.samples_len = len(self.texts)
        self.max_len = max_len
        self.index_to_char = index_to_char

    # 获取样本的条数
    def __len__(self):
        return self.samples_len

    def __getitem__(self, idx):
        idx = min(max(idx, 0), self.samples_len - 1)
        # 根据索引取出样本
        x = self.texts[idx]
        print(self.texts[idx],"X")
        y = self.string_labels[idx]
        print("Y", self.string_labels[idx])
        # 创建张量
        indices = [self.index_to_char.get(char, 0) for char in x[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))

        tensor = torch.LongTensor(indices)
        return torch.tensor(indices, dtype=torch.long), self.string_labels[idx]


def get_dataloader():
    texts, string_labels,index_to_char, max_len = read_data("dataset.csv")
    name_dataset = NameDataset(texts, string_labels, index_to_char, max_len)
    # 封装dataset 得到dataloader对象：会对数据进行增维
    train_dataloader = DataLoader(name_dataset, batch_size=1, shuffle=True)
    return train_dataloader


class NameRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(self.input_size, self.hidden_size, num_layers)
        # 定义输出层
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # 定义logSoftmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h0):
        print(f"x->{x.shape}")
        print(f"h0->{h0.shape}")
        # ho 初始化的值
        # X身维
        x1 = torch.unsqueeze(x, dim=1)  # x.unsqueeze(dim=1)
        print(f"x1->{x1.shape}")
        output, hn = self.rnn(x1, h0)
        print(f"output: {output}")
        print(f"hn: {hn}")

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


def classify_text_run():
    train_dataloader = get_dataloader()
    # for tensor_x, tensor_y in train_dataloader:
    # print(f"tensor_x--->{tensor_x}")
    # print(f"tensor_x--->{tensor_x.shape}")
    # print(f"tensor_y--->{tensor_y}")
    #  break
    model = NameRNN(input_size=57, hidden_size=128, output_size=18)
    print(model, "rnn")
    h0 = model.initHidden()
    for x, y in train_dataloader:
        print(x[0], "X0", x.shape)
        out_put, hn = model(x[0], h0)


class NameLstm(nn.Module):
    pass

class NameGRU(nn.Module):
    pass

def test_dataset():
    texts, string_labels,index_to_char, max_len = read_data("dataset.csv")
    name_dataset = NameDataset(texts, string_labels, index_to_char, max_len)
    print(len(name_dataset))
    print(name_dataset.__getitem__(0))



if __name__ == '__main__':
    train_dataloader = get_dataloader()
    # for tensor_x, tensor_y in train_dataloader:
        # print(f"tensor_x--->{tensor_x}")
       # print(f"tensor_x--->{tensor_x.shape}")
         # print(f"tensor_y--->{tensor_y}")
       #  break
    model = NameRNN(input_size=57, hidden_size=128, output_size=18)
    print(model, "rnn")
    h0 = model.initHidden()
    for x, y in train_dataloader:
        print(x[0],"X0",x.shape)
        out_put, hn = model(x[0], h0)
        #break
