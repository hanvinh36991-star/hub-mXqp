
import torch
import torch.nn as nn 
from torch.optim import Adam, SGD
from transformers import BertModel

"""
建立网络结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCnn(config)
        elif model_type == "bert":
            self.use_bert = True 
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False) # return_dict=False → 返回 tuple
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size #self.encoder 之所以有 .bert 这个属性，是因为在 BertMidLayer.__init__ 里把它“挂”上去了
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x, attention_mask=None, target = None):
        if self.use_bert:
            x = self.encoder(x, attention_mask=attention_mask)    # bert返回的结果 (last_hidden_state, pooler_output)格式 （B,L,H） (B, H)
            # return_dict=False -> (last_hidden_state, pooler_output)
            if isinstance(x, tuple):
                x = x[0]  # (B, L, H)   
        else:
            emb = self.embedding(x)  # (B, L, H)
            x = self.encoder(emb)    # (B, L, H)

            if isinstance(x, tuple): #  # RNN/LSTM/GRU 类的模型同时会返回隐单元向量
                x = x[0]

        # if self.pooling_style == "max":
        #     self.pooling_layer = nn.MaxPool1d(x.shape[1])
        #     #self.pooling_layer = nn.AdaptiveMaxPool1d(1) #option2: 1指的是压缩成长度等于1
        # else:
        #     self.pooling_layer = nn.AvgPool1d(x.shape[1])
        
        # x = self.pooling_layer(x.transpose(1,2)).squeeze(-1) # (B,L,H)->(B,H,L)-> (B,H)
        if self.pooling_style == "max":
            x = x.max(dim=1).values         # torch中max返回（values,indices）
        else: 
            x = x.mean(dim=1)    # (B, H)
        #也可以直接使用序列[cls]一个位置的向量
        # x = x[:, 1, :]
        predict = self.classify(x)  # (B, C)
        if target is not None:
            return self.loss(predict, target.squeeze())  # target如果是二维应该压成1维 成 (B,)
        else:
            return predict 
        
class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1)/2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias = False, padding=pad)
    
    def forward(self, x): # (B, L, N)
        return self.cnn(x.transpose(1,2)).transpose(1,2)

class GatedCnn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)
    
    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)

class BertMidLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict = False)
        self.bert.config.output_hidden_states = True
        '''
             (
            last_hidden_state,      # index 0, (B, L, H)
            pooler_output,          # index 1, (B, H)
            hidden_states           # index 2, tuple of layers
            )

        '''
    
    def forward(self, x, attention_mask=None):
        layer_states = self.bert(x,attention_mask=attention_mask)[2] # (13, B, L, H)  (0是embedding, 1~12是每层transformer)
        #最后一层：偏向任务、偏线性、偏“被 pretrain objective 拉歪”倒数第二层：语更稳定、更通用
        layer_states = torch.add(layer_states[-2], layer_states[-1]) 
        return layer_states # (B, L, H)

class BertLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict = False)
        hidden_size = self.bert.config.hidden_size
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
    
    def forward(self, x,attention_mask=None):
        x = self.bert(x,attention_mask=attention_mask)[0]
        x, _ =  self.rnn(x)
        return x

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(),lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    # 这里一般是测试用例
    from config import Config

    Config["model_type"] = "bert"
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[0,1,2,3,4], [5,6,7,8,9], [4,5,3,4,2]])  # (B, L)
    print(x.shape) # (2,5)
    sequence_output, pooler_output = model(x)
    print(x[2], type(x[2]), len(x[2]))
    print(sequence_output.shape)