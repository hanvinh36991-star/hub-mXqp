import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

# 构建语料库
sentences = [
    ['毛老师 喜欢 人工智能', '<sos> TeacherMao likes AI', 'TeacherMao likes AI <eos>'],
    ['我 爱 学习 世界', '<sos> I love studying world', 'I love studying world <eos>'],
    ['深度学习 改变 世界', '<sos> DL changed the world', 'DL changed the world <eos>'],
    ['自然 语言 处理 很 强大', '<sos> NLP is so powerful', 'NLP is so powerful <eos>'],
    ['神经网络 非常 复杂', '<sos> Neural-Nets are complex', 'Neural-Nets are complex <eos>']
]

# 构建词汇表
word_list_cn, word_list_en = [], []
for s in sentences:
    word_list_cn.extend(s[0].split())
    word_list_en.extend(s[1].split())
    word_list_en.extend(s[2].split())

# 添加特殊标记
special_tokens_en = ['<pad>', '<unk>', '<sos>', '<eos>']
special_tokens_cn = ['<pad>', '<unk>']

word_list_cn = special_tokens_cn + list(set(word_list_cn))
word_list_en = special_tokens_en + list(set(word_list_en))

# 构建映射字典
word2idx_cn = {w: i for i, w in enumerate(word_list_cn)}
word2idx_en = {w: i for i, w in enumerate(word_list_en)}
idx2word_cn = {i: w for i, w in enumerate(word_list_cn)}
idx2word_en = {i: w for i, w in enumerate(word_list_en)}

voc_size_cn = len(word_list_cn)
voc_size_en = len(word_list_en)


# 定义Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden


# 定义Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output)
        return output, hidden


# 定义Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input, hidden, dec_input, teacher_forcing_ratio=0.5):
        # Encoder前向传播
        encoder_output, encoder_hidden = self.encoder(enc_input, hidden)

        # Decoder前向传播（训练时使用teacher forcing）
        batch_size = enc_input.size(0)
        target_len = dec_input.size(1)

        # 存储所有时间步的输出
        decoder_outputs = torch.zeros(batch_size, target_len, voc_size_en)

        decoder_input = dec_input[:, 0].unsqueeze(1)  # 第一个输入是<sos>
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: 使用真实标签作为下一个输入
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_outputs[:, t, :] = decoder_output.squeeze(1)
                decoder_input = dec_input[:, t].unsqueeze(1) if t < target_len - 1 else None
        else:
            # 不使用teacher forcing: 使用预测值作为下一个输入
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_outputs[:, t, :] = decoder_output.squeeze(1)

                # 获取预测值
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        return decoder_outputs


# 数据准备函数
def make_data(sentences, max_len=10):
    random_sentence = random.choice(sentences)

    # 编码器输入
    encoder_words = random_sentence[0].split()
    encoder_input_idx = [word2idx_cn.get(w, word2idx_cn['<unk>']) for w in encoder_words]
    encoder_input_idx = encoder_input_idx[:max_len] + [word2idx_cn['<pad>']] * (max_len - len(encoder_input_idx))

    # 解码器输入（包含<sos>）
    decoder_words = random_sentence[1].split()
    decoder_input_idx = [word2idx_en.get(w, word2idx_en['<unk>']) for w in decoder_words]
    decoder_input_idx = decoder_input_idx[:max_len] + [word2idx_en['<pad>']] * (max_len - len(decoder_input_idx))

    # 目标输出（包含<eos>）
    target_words = random_sentence[2].split()
    target_idx = [word2idx_en.get(w, word2idx_en['<unk>']) for w in target_words]
    target_idx = target_idx[:max_len] + [word2idx_en['<pad>']] * (max_len - len(target_idx))

    # 转换为张量
    encoder_input = torch.LongTensor([encoder_input_idx])
    decoder_input = torch.LongTensor([decoder_input_idx])
    target = torch.LongTensor([target_idx])

    return encoder_input, decoder_input, target


# 创建模型
n_hidden = 128
encoder = Encoder(voc_size_cn, n_hidden)
decoder = Decoder(n_hidden, voc_size_en)
model = Seq2Seq(encoder, decoder)

print(f"模型结构: {model}")
print(f"中文词汇表大小: {voc_size_cn}")
print(f"英文词汇表大小: {voc_size_en}")


# 训练函数
def train_seq2seq(model, criterion, optimizer, epochs):
    model.train()

    for epoch in range(epochs):
        # 准备数据
        encoder_input, decoder_input, target = make_data(sentences)

        # 初始化隐藏状态
        hidden = torch.zeros(1, encoder_input.size(0), n_hidden)

        # 前向传播
        optimizer.zero_grad()
        output = model(encoder_input, hidden, decoder_input, teacher_forcing_ratio=0.5)

        # 计算损失
        loss = criterion(output.view(-1, voc_size_en), target.view(-1))

        # 反向传播
        loss.backward()
        optimizer.step()

        # 打印训练信息
        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch + 1:04d}, Loss: {loss.item():.6f}")


# 训练模型
print("\n开始训练模型...")
epochs = 2000
criterion = nn.CrossEntropyLoss(ignore_index=word2idx_en['<pad>'])  # 忽略padding的损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_seq2seq(model, criterion, optimizer, epochs)


# 测试函数
def translate(model, source_sentence, max_len=15):
    model.eval()

    with torch.no_grad():
        # 处理输入句子
        words = source_sentence.split()
        encoder_input_idx = [word2idx_cn.get(w, word2idx_cn['<unk>']) for w in words]

        # 填充或截断
        encoder_input_idx = encoder_input_idx[:max_len] + [word2idx_cn['<pad>']] * (max_len - len(encoder_input_idx))
        encoder_input = torch.LongTensor([encoder_input_idx])

        # Encoder编码
        hidden = torch.zeros(1, encoder_input.size(0), n_hidden)
        encoder_output, encoder_hidden = encoder(encoder_input, hidden)

        # 开始解码
        decoder_hidden = encoder_hidden
        decoder_input = torch.LongTensor([[word2idx_en['<sos>']]])

        translated_words = []

        for i in range(max_len):
            # Decoder解码
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            # 获取预测词
            topv, topi = decoder_output.data.topk(1)
            next_word_idx = topi.squeeze().item()
            next_word = idx2word_en[next_word_idx]

            # 遇到<eos>则停止
            if next_word == '<eos>':
                break

            # 添加到翻译结果
            if next_word != '<sos>' and next_word != '<pad>':
                translated_words.append(next_word)

            # 更新输入
            decoder_input = torch.LongTensor([[next_word_idx]])

        return ' '.join(translated_words)


# 测试模型
print("\n" + "=" * 50)
print("翻译测试结果:")
print("=" * 50)

test_sentences = [
    '毛老师 喜欢 人工智能',
    '我 爱 学习 世界',
    '深度学习 改变 世界',
    '自然 语言 处理 很 强大',
    '神经网络 非常 复杂'
]

for test_sent in test_sentences:
    translation = translate(model, test_sent)
    print(f"中文: {test_sent}")
    print(f"英文: {translation}")
    print("-" * 40)

# 尝试一些不在训练集中的句子（但词汇在词表中）
print("\n" + "=" * 50)
print("额外测试:")
print("=" * 50)

extra_tests = [
    '人工智能 很 有趣',
    '机器学习 改变 生活'
]

for test_sent in extra_tests:
    # 检查是否所有词都在词表中
    unknown_words = [w for w in test_sent.split() if w not in word2idx_cn]
    if unknown_words:
        print(f"跳过 '{test_sent}' - 包含未知词: {unknown_words}")
    else:
        translation = translate(model, test_sent)
        print(f"中文: {test_sent}")
        print(f"英文: {translation}")
        print("-" * 40)

# 批量翻译示例
print("\n" + "=" * 50)
print("批量翻译演示:")
print("=" * 50)

batch_sentences = ['毛老师 喜欢 人工智能', '我 爱 学习 世界']
for i, sent in enumerate(batch_sentences):
    translation = translate(model, sent)
    print(f"句子 {i + 1}: {sent}")
    print(f"翻译: {translation}\n")

print("模型训练和测试完成!")