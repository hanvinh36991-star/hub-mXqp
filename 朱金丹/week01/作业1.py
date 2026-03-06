#coding:utf8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import jieba
import random

# ===================== 全局配置 =====================
# 随机种子（保证结果可复现）
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# 设备配置（优先使用GPU）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据配置
MAX_LENGTH = 128  # 文本最大长度
BATCH_SIZE = 32   # 批次大小
EMBEDDING_DIM = 128  # 词嵌入维度
VOCAB_SIZE = 5000  # 词汇表大小（取出现频率前5000的词）

# 训练配置
EPOCHS = 10
LEARNING_RATE = 1e-3

# ===================== 数据预处理 =====================
class TextDataset(Dataset):
    """自定义文本数据集类"""
    def __init__(self, texts, labels, word2idx, max_length):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 文本转序列（分词+ID映射+padding/truncation）
        text = self.texts[idx]
        words = jieba.lcut(text)[:self.max_length]  # 分词并截断
        # 词转ID（未知词用UNK，ID=1）
        seq = [self.word2idx.get(word, 1) for word in words]
        # padding（不足max_length补0）
        seq += [0] * (self.max_length - len(seq))
        seq = torch.LongTensor(seq)
        
        # 标签转张量
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return seq, label

def load_and_preprocess_data(csv_path):
    """加载并预处理数据"""
    # 1. 读取CSV
    df = pd.read_csv(csv_path)
    # 检查必要列
    assert 'text' in df.columns and 'label' in df.columns, "CSV需包含text和label列"
    
    # 2. 标签编码（将字符串标签转为数字）
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    num_classes = len(le.classes_)  # 分类类别数
    
    # 3. 构建词汇表（基于训练集）
    # 先划分训练集和测试集（8:2）
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), 
        test_size=0.2, random_state=SEED, stratify=df['label']
    )
    
    # 统计词频
    word_freq = {}
    for text in train_texts:
        words = jieba.lcut(text)
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # 构建词表（按词频排序，取前VOCAB_SIZE个）
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:VOCAB_SIZE-2]
    word2idx = {
        '<PAD>': 0,  # padding标记
        '<UNK>': 1   # 未知词标记
    }
    for idx, (word, _) in enumerate(sorted_words, 2):
        word2idx[word] = idx
    
    # 4. 构建数据集
    train_dataset = TextDataset(train_texts, train_labels, word2idx, MAX_LENGTH)
    test_dataset = TextDataset(test_texts, test_labels, word2idx, MAX_LENGTH)
    
    # 5. 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, word2idx, num_classes, le

# ===================== 模型1：CNN文本分类模型 =====================
class TextCNN(nn.Module):
    """
    经典TextCNN模型：嵌入层 + 多尺度卷积 + 池化 + 全连接
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, max_length):
        super(TextCNN, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # 多尺度卷积层（3/4/5-gram）
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,  # 通道数（文本为1维）
                out_channels=128,  # 卷积核数量
                kernel_size=(k, embedding_dim)  # 卷积核大小（窗口大小×嵌入维度）
            ) for k in [3, 4, 5]
        ])
        
        # 激活函数
        self.relu = nn.ReLU()
        
        # 池化层（全局最大池化）
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # 全连接层（分类）
        self.fc = nn.Linear(128 * 3, num_classes)  # 3个卷积核×128输出
        
        # Dropout（防止过拟合）
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [batch_size, max_length] → 输入序列
        
        # 1. 嵌入层：[batch_size, max_length] → [batch_size, max_length, embedding_dim]
        x = self.embedding(x)
        
        # 2. 增加通道维度：[batch_size, max_length, embedding_dim] → [batch_size, 1, max_length, embedding_dim]
        x = x.unsqueeze(1)
        
        # 3. 多尺度卷积+池化
        conv_outs = []
        for conv in self.convs:
            # 卷积：[batch_size, 1, max_length, embedding_dim] → [batch_size, 128, max_length-k+1, 1]
            out = conv(x)
            # 去掉最后一维：[batch_size, 128, max_length-k+1]
            out = out.squeeze(3)
            # 激活
            out = self.relu(out)
            # 池化：[batch_size, 128, max_length-k+1] → [batch_size, 128, 1]
            out = self.pool(out)
            # 去掉最后一维：[batch_size, 128]
            out = out.squeeze(2)
            conv_outs.append(out)
        
        # 4. 拼接多尺度特征：[batch_size, 128*3]
        concat = torch.cat(conv_outs, dim=1)
        
        # 5. Dropout
        concat = self.dropout(concat)
        
        # 6. 分类：[batch_size, 128*3] → [batch_size, num_classes]
        logits = self.fc(concat)
        
        return logits

# ===================== 模型2：BiLSTM文本分类模型 =====================
class TextBiLSTM(nn.Module):
    """
    BiLSTM模型：嵌入层 + 双向LSTM + 全连接
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, hidden_size=128, num_layers=2):
        super(TextBiLSTM, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,  # 双向
            batch_first=True,    # 输入形状：[batch_size, seq_len, feature]
            dropout=0.5 if num_layers > 1 else 0  # 多层时使用dropout
        )
        
        # 全连接层（分类）：双向LSTM输出维度=2*hidden_size
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [batch_size, max_length] → 输入序列
        
        # 1. 嵌入层：[batch_size, max_length] → [batch_size, max_length, embedding_dim]
        x = self.embedding(x)
        
        # 2. LSTM层：[batch_size, max_length, embedding_dim] → [batch_size, max_length, 2*hidden_size]
        # _: (h_n, c_n) 隐藏状态和细胞状态，未使用
        out, _ = self.lstm(x)
        
        # 3. 取最后一个时间步的输出：[batch_size, 2*hidden_size]
        out = out[:, -1, :]
        
        # 4. Dropout
        out = self.dropout(out)
        
        # 5. 分类：[batch_size, 2*hidden_size] → [batch_size, num_classes]
        logits = self.fc(out)
        
        return logits

# ===================== 训练与评估函数 =====================
def train_model(model, train_loader, criterion, optimizer, epoch):
    """训练单个epoch"""
    model.train()
    total_loss = 0.0
    for batch_idx, (seqs, labels) in enumerate(train_loader):
        seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
        
        # 梯度归零
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(seqs)
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item()
        
        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch {epoch+1} | Batch {batch_idx+1} | Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1} | Train Avg Loss: {avg_loss:.4f}')
    return avg_loss

def evaluate_model(model, test_loader, criterion, label_encoder):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # 禁用梯度计算
        for seqs, labels in test_loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            
            # 前向传播
            logits = model(seqs)
            
            # 计算损失
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # 预测类别（取概率最大的类别）
            preds = torch.argmax(logits, dim=1)
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(test_loader)
    
    # 打印分类报告
    print("\n分类报告：")
    print(classification_report(
        all_labels, all_preds,
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    return avg_loss, all_preds, all_labels

# ===================== 主函数 =====================
def main(csv_path="dataset.csv"):
    # 1. 数据预处理
    print("===== 加载并预处理数据 =====")
    train_loader, test_loader, word2idx, num_classes, le = load_and_preprocess_data(csv_path)
    print(f"词汇表大小：{len(word2idx)}")
    print(f"分类类别数：{num_classes}")
    print(f"训练集批次：{len(train_loader)} | 测试集批次：{len(test_loader)}")
    
    # 2. 定义模型、损失函数、优化器
    # 模型1：TextCNN
    print("\n===== 训练TextCNN模型 =====")
    cnn_model = TextCNN(
        vocab_size=len(word2idx),
        embedding_dim=EMBEDDING_DIM,
        num_classes=num_classes,
        max_length=MAX_LENGTH
    ).to(DEVICE)
    
    cnn_criterion = nn.CrossEntropyLoss()
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    
    # 训练CNN
    for epoch in range(EPOCHS):
        train_model(cnn_model, train_loader, cnn_criterion, cnn_optimizer, epoch)
    
    # 评估CNN
    print("\n===== 评估TextCNN模型 =====")
    cnn_test_loss, _, _ = evaluate_model(cnn_model, test_loader, cnn_criterion, le)
    print(f"TextCNN Test Avg Loss: {cnn_test_loss:.4f}")
    
    # 保存CNN模型
    torch.save(cnn_model.state_dict(), "text_cnn_model.pth")
    print("TextCNN模型已保存为 text_cnn_model.pth")
    
    # 模型2：TextBiLSTM
    print("\n===== 训练TextBiLSTM模型 =====")
    bilstm_model = TextBiLSTM(
        vocab_size=len(word2idx),
        embedding_dim=EMBEDDING_DIM,
        num_classes=num_classes,
        hidden_size=128,
        num_layers=2
    ).to(DEVICE)
    
    bilstm_criterion = nn.CrossEntropyLoss()
    bilstm_optimizer = optim.Adam(bilstm_model.parameters(), lr=LEARNING_RATE)
    
    # 训练BiLSTM
    for epoch in range(EPOCHS):
        train_model(bilstm_model, train_loader, bilstm_criterion, bilstm_optimizer, epoch)
    
    # 评估BiLSTM
    print("\n===== 评估TextBiLSTM模型 =====")
    bilstm_test_loss, _, _ = evaluate_model(bilstm_model, test_loader, bilstm_criterion, le)
    print(f"TextBiLSTM Test Avg Loss: {bilstm_test_loss:.4f}")
    
    # 保存BiLSTM模型
    torch.save(bilstm_model.state_dict(), "text_bilstm_model.pth")
    print("TextBiLSTM模型已保存为 text_bilstm_model.pth")
    
    # 对比结果
    print("\n===== 模型对比 =====")
    print(f"TextCNN 测试损失：{cnn_test_loss:.4f}")
    print(f"TextBiLSTM 测试损失：{bilstm_test_loss:.4f}")
    print("注：损失越低、分类报告中F1值越高，模型效果越好")

if __name__ == "__main__":
    # 替换为你的dataset.csv路径
    main(csv_path="dataset.csv")
