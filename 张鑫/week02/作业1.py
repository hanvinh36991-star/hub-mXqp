import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
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


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 调整方法 1：增加节点个数 (更宽的网络) - 512个节点
class WiderClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WiderClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512) # 从128增加到512
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 调整方法 2：增加层的个数 (更深的网络) - 2个隐藏层
class DeeperClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeeperClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # 增加一层
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

hidden_dim = 128
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, output_dim) # 维度和精度有什么关系？
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.SGD(model.parameters(), lr=0.01)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

num_epochs = 10
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
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


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

print("\n" + "="*30)
print("开始对比实验")
print("="*30)


# --- 对比方法 1: 调整节点个数 (将 hidden_dim 从 128 改为 512) ---
print("\n--- 实验 1: 增加节点个数 (hidden_dim=512) ---")
model_wide = SimpleClassifier(vocab_size, 512, output_dim)
optimizer_wide = optim.SGD(model_wide.parameters(), lr=0.01)

for epoch in range(num_epochs):
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer_wide.zero_grad()
        outputs = model_wide(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_wide.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"实验1 Batch {idx}, Loss: {loss.item():.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# --- 对比方法 2: 调整层的个数 (增加一个隐藏层) ---
print("\n--- 实验 2: 增加层的个数 (深度增加) ---")

class DeeperClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeeperClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 新增的一层
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

model_deep = DeeperClassifier(vocab_size, 128, output_dim)
optimizer_deep = optim.SGD(model_deep.parameters(), lr=0.01)

for epoch in range(num_epochs):
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer_deep.zero_grad()
        outputs = model_deep(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_deep.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"实验2 Batch {idx}, Loss: {loss.item():.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# --- 对比方法 3: 换用 Adam 优化器 (基础模型, hidden_dim=128) ---
print("\n--- 实验 3: 换用 Adam 优化器 (基础模型, lr=0.01) ---")
model_adam = SimpleClassifier(vocab_size, 128, output_dim)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.01)

for epoch in range(num_epochs):
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer_adam.zero_grad()
        outputs = model_adam(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_adam.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"实验3 Batch {idx}, Loss: {loss.item():.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


print("\n对比验证完成。")


"""
结果：
  import pynvml  # type: ignore[import]
Batch 个数 0, 当前Batch Loss: 2.4882218837738037
Batch 个数 50, 当前Batch Loss: 2.475986957550049
Batch 个数 100, 当前Batch Loss: 2.4545881748199463
Batch 个数 150, 当前Batch Loss: 2.436913013458252
Batch 个数 200, 当前Batch Loss: 2.4261343479156494
Batch 个数 250, 当前Batch Loss: 2.377124786376953
Batch 个数 300, 当前Batch Loss: 2.3638150691986084
Batch 个数 350, 当前Batch Loss: 2.3267557621002197
Epoch [1/10], Loss: 2.4170
Batch 个数 0, 当前Batch Loss: 2.3037819862365723
Batch 个数 50, 当前Batch Loss: 2.3703465461730957
Batch 个数 100, 当前Batch Loss: 2.3094568252563477
Batch 个数 150, 当前Batch Loss: 2.215402364730835
Batch 个数 200, 当前Batch Loss: 2.1908884048461914
Batch 个数 250, 当前Batch Loss: 2.195530414581299
Batch 个数 300, 当前Batch Loss: 2.088827133178711
Batch 个数 350, 当前Batch Loss: 2.169043779373169
Epoch [2/10], Loss: 2.2206
Batch 个数 0, 当前Batch Loss: 2.0684518814086914
Batch 个数 50, 当前Batch Loss: 2.018367290496826
Batch 个数 100, 当前Batch Loss: 2.003796339035034
Batch 个数 150, 当前Batch Loss: 1.9581471681594849
Batch 个数 200, 当前Batch Loss: 1.9585076570510864
Batch 个数 250, 当前Batch Loss: 1.7606819868087769
Batch 个数 300, 当前Batch Loss: 1.787622094154358
Batch 个数 350, 当前Batch Loss: 1.9662753343582153
Epoch [3/10], Loss: 1.9262
Batch 个数 0, 当前Batch Loss: 1.8696722984313965
Batch 个数 50, 当前Batch Loss: 1.585164189338684
Batch 个数 100, 当前Batch Loss: 1.5553323030471802
Batch 个数 150, 当前Batch Loss: 1.5629202127456665
Batch 个数 200, 当前Batch Loss: 1.599893569946289
Batch 个数 250, 当前Batch Loss: 1.4902068376541138
Batch 个数 300, 当前Batch Loss: 1.490405797958374
Batch 个数 350, 当前Batch Loss: 1.4861723184585571
Epoch [4/10], Loss: 1.5554
Batch 个数 0, 当前Batch Loss: 1.5271168947219849
Batch 个数 50, 当前Batch Loss: 1.2773698568344116
Batch 个数 100, 当前Batch Loss: 1.2005773782730103
Batch 个数 150, 当前Batch Loss: 1.3593069314956665
Batch 个数 200, 当前Batch Loss: 1.3653053045272827
Batch 个数 250, 当前Batch Loss: 1.1950469017028809
Batch 个数 300, 当前Batch Loss: 1.357407569885254
Batch 个数 350, 当前Batch Loss: 0.8611066341400146
Epoch [5/10], Loss: 1.2186
Batch 个数 0, 当前Batch Loss: 1.0606242418289185
Batch 个数 50, 当前Batch Loss: 0.8606799244880676
Batch 个数 100, 当前Batch Loss: 0.7102088928222656
Batch 个数 150, 当前Batch Loss: 1.004323959350586
Batch 个数 200, 当前Batch Loss: 0.7634719610214233
Batch 个数 250, 当前Batch Loss: 0.8530341982841492
Batch 个数 300, 当前Batch Loss: 0.7890105247497559
Batch 个数 350, 当前Batch Loss: 0.6719601154327393
Epoch [6/10], Loss: 0.9819
Batch 个数 0, 当前Batch Loss: 0.9354149103164673
Batch 个数 50, 当前Batch Loss: 0.9587666392326355
Batch 个数 100, 当前Batch Loss: 0.8631629347801208
Batch 个数 150, 当前Batch Loss: 0.8713394999504089
Batch 个数 200, 当前Batch Loss: 0.6810790300369263
Batch 个数 250, 当前Batch Loss: 0.6325746774673462
Batch 个数 300, 当前Batch Loss: 0.5296482443809509
Batch 个数 350, 当前Batch Loss: 0.7829822301864624
Epoch [7/10], Loss: 0.8252
Batch 个数 0, 当前Batch Loss: 0.5702527761459351
Batch 个数 50, 当前Batch Loss: 0.7207492589950562
Batch 个数 100, 当前Batch Loss: 0.489292174577713
Batch 个数 150, 当前Batch Loss: 0.7794771194458008
Batch 个数 200, 当前Batch Loss: 0.7373847365379333
Batch 个数 250, 当前Batch Loss: 1.0593892335891724
Batch 个数 300, 当前Batch Loss: 0.6860949397087097
Batch 个数 350, 当前Batch Loss: 0.8730605244636536
Epoch [8/10], Loss: 0.7170
Batch 个数 0, 当前Batch Loss: 0.8975726962089539
Batch 个数 50, 当前Batch Loss: 0.5708820819854736
Batch 个数 100, 当前Batch Loss: 0.8866637349128723
Batch 个数 150, 当前Batch Loss: 0.41101598739624023
Batch 个数 200, 当前Batch Loss: 0.7286128997802734
Batch 个数 250, 当前Batch Loss: 0.8352553844451904
Batch 个数 300, 当前Batch Loss: 0.3257867097854614
Batch 个数 350, 当前Batch Loss: 0.3297145664691925
Epoch [9/10], Loss: 0.6449
Batch 个数 0, 当前Batch Loss: 0.5170977115631104
Batch 个数 50, 当前Batch Loss: 0.4450100064277649
Batch 个数 100, 当前Batch Loss: 0.37570783495903015
Batch 个数 150, 当前Batch Loss: 0.5418042540550232
Batch 个数 200, 当前Batch Loss: 0.5557698607444763
Batch 个数 250, 当前Batch Loss: 0.7334647178649902
Batch 个数 300, 当前Batch Loss: 0.30930739641189575
Batch 个数 350, 当前Batch Loss: 0.5124892592430115
Epoch [10/10], Loss: 0.5878
输入 '帮我导航到北京' 预测为: 'Travel-Query'
输入 '查询明天北京的天气' 预测为: 'Weather-Query'

==============================
开始对比实验
==============================

--- 实验 1: 增加节点个数 (hidden_dim=512) ---
实验1 Batch 0, Loss: 2.4866
实验1 Batch 50, Loss: 2.4672
实验1 Batch 100, Loss: 2.4640
实验1 Batch 150, Loss: 2.4124
实验1 Batch 200, Loss: 2.4252
实验1 Batch 250, Loss: 2.3712
实验1 Batch 300, Loss: 2.3699
实验1 Batch 350, Loss: 2.3465
Epoch [1/10], Loss: 2.4086
实验1 Batch 0, Loss: 2.3043
实验1 Batch 50, Loss: 2.3003
实验1 Batch 100, Loss: 2.2680
实验1 Batch 150, Loss: 2.1906
实验1 Batch 200, Loss: 2.2703
实验1 Batch 250, Loss: 2.2585
实验1 Batch 300, Loss: 2.1038
实验1 Batch 350, Loss: 2.0808
Epoch [2/10], Loss: 2.2097
实验1 Batch 0, Loss: 2.0705
实验1 Batch 50, Loss: 1.9295
实验1 Batch 100, Loss: 1.9364
实验1 Batch 150, Loss: 1.8990
实验1 Batch 200, Loss: 1.7509
实验1 Batch 250, Loss: 1.7944
实验1 Batch 300, Loss: 1.8080
实验1 Batch 350, Loss: 1.8125
Epoch [3/10], Loss: 1.9005
实验1 Batch 0, Loss: 1.7350
实验1 Batch 50, Loss: 1.5099
实验1 Batch 100, Loss: 1.7912
实验1 Batch 150, Loss: 1.6428
实验1 Batch 200, Loss: 1.3820
实验1 Batch 250, Loss: 1.2915
实验1 Batch 300, Loss: 1.3966
实验1 Batch 350, Loss: 1.3525
Epoch [4/10], Loss: 1.5233
实验1 Batch 0, Loss: 1.2524
实验1 Batch 50, Loss: 1.2642
实验1 Batch 100, Loss: 1.1724
实验1 Batch 150, Loss: 1.1728
实验1 Batch 200, Loss: 1.2707
实验1 Batch 250, Loss: 1.0709
实验1 Batch 300, Loss: 1.2240
实验1 Batch 350, Loss: 1.2152
Epoch [5/10], Loss: 1.1982
实验1 Batch 0, Loss: 1.3213
实验1 Batch 50, Loss: 0.9448
实验1 Batch 100, Loss: 1.1484
实验1 Batch 150, Loss: 1.0255
实验1 Batch 200, Loss: 0.9761
实验1 Batch 250, Loss: 1.0212
实验1 Batch 300, Loss: 0.8308
实验1 Batch 350, Loss: 0.9871
Epoch [6/10], Loss: 0.9702
实验1 Batch 0, Loss: 1.1234
实验1 Batch 50, Loss: 0.9015
实验1 Batch 100, Loss: 0.7044
实验1 Batch 150, Loss: 0.5762
实验1 Batch 200, Loss: 0.7542
实验1 Batch 250, Loss: 0.8465
实验1 Batch 300, Loss: 0.9712
实验1 Batch 350, Loss: 0.5643
Epoch [7/10], Loss: 0.8169
实验1 Batch 0, Loss: 0.8335
实验1 Batch 50, Loss: 0.6338
实验1 Batch 100, Loss: 0.6663
实验1 Batch 150, Loss: 0.4462
实验1 Batch 200, Loss: 0.7666
实验1 Batch 250, Loss: 0.8668
实验1 Batch 300, Loss: 0.7813
实验1 Batch 350, Loss: 0.8191
Epoch [8/10], Loss: 0.7128
实验1 Batch 0, Loss: 0.7318
实验1 Batch 50, Loss: 0.9173
实验1 Batch 100, Loss: 0.5169
实验1 Batch 150, Loss: 0.8120
实验1 Batch 200, Loss: 0.7117
实验1 Batch 250, Loss: 0.5777
实验1 Batch 300, Loss: 0.6895
实验1 Batch 350, Loss: 0.7132
Epoch [9/10], Loss: 0.6380
实验1 Batch 0, Loss: 0.7008
实验1 Batch 50, Loss: 0.6842
实验1 Batch 100, Loss: 0.4229
实验1 Batch 150, Loss: 0.6773
实验1 Batch 200, Loss: 0.3904
实验1 Batch 250, Loss: 0.9063
实验1 Batch 300, Loss: 0.6012
实验1 Batch 350, Loss: 0.8076
Epoch [10/10], Loss: 0.5833

--- 实验 2: 增加层的个数 (深度增加) ---
实验2 Batch 0, Loss: 2.4794
实验2 Batch 50, Loss: 2.4755
实验2 Batch 100, Loss: 2.4542
实验2 Batch 150, Loss: 2.4401
实验2 Batch 200, Loss: 2.4396
实验2 Batch 250, Loss: 2.4226
实验2 Batch 300, Loss: 2.4230
实验2 Batch 350, Loss: 2.4069
Epoch [1/10], Loss: 2.4467
实验2 Batch 0, Loss: 2.3910
实验2 Batch 50, Loss: 2.4395
实验2 Batch 100, Loss: 2.3861
实验2 Batch 150, Loss: 2.4281
实验2 Batch 200, Loss: 2.3841
实验2 Batch 250, Loss: 2.3832
实验2 Batch 300, Loss: 2.3542
实验2 Batch 350, Loss: 2.3762
Epoch [2/10], Loss: 2.3891
实验2 Batch 0, Loss: 2.4191
实验2 Batch 50, Loss: 2.3910
实验2 Batch 100, Loss: 2.2851
实验2 Batch 150, Loss: 2.3149
实验2 Batch 200, Loss: 2.3734
实验2 Batch 250, Loss: 2.2619
实验2 Batch 300, Loss: 2.2585
实验2 Batch 350, Loss: 2.2914
Epoch [3/10], Loss: 2.3283
实验2 Batch 0, Loss: 2.2657
实验2 Batch 50, Loss: 2.2198
实验2 Batch 100, Loss: 2.2545
实验2 Batch 150, Loss: 2.2956
实验2 Batch 200, Loss: 2.1815
实验2 Batch 250, Loss: 2.1149
实验2 Batch 300, Loss: 2.0986
实验2 Batch 350, Loss: 2.0996
Epoch [4/10], Loss: 2.2158
实验2 Batch 0, Loss: 2.0649
实验2 Batch 50, Loss: 2.1072
实验2 Batch 100, Loss: 2.1273
实验2 Batch 150, Loss: 2.0250
实验2 Batch 200, Loss: 1.8617
实验2 Batch 250, Loss: 1.9556
实验2 Batch 300, Loss: 1.7058
实验2 Batch 350, Loss: 1.8630
Epoch [5/10], Loss: 1.9490
实验2 Batch 0, Loss: 1.5732
实验2 Batch 50, Loss: 1.9596
实验2 Batch 100, Loss: 1.5363
实验2 Batch 150, Loss: 1.5061
实验2 Batch 200, Loss: 1.6463
实验2 Batch 250, Loss: 1.4378
实验2 Batch 300, Loss: 1.3712
实验2 Batch 350, Loss: 1.5022
Epoch [6/10], Loss: 1.5606
实验2 Batch 0, Loss: 1.5110
实验2 Batch 50, Loss: 1.5251
实验2 Batch 100, Loss: 1.4809
实验2 Batch 150, Loss: 1.4401
实验2 Batch 200, Loss: 1.4728
实验2 Batch 250, Loss: 1.1391
实验2 Batch 300, Loss: 1.1474
实验2 Batch 350, Loss: 0.9259
Epoch [7/10], Loss: 1.2062
实验2 Batch 0, Loss: 1.2599
实验2 Batch 50, Loss: 0.7887
实验2 Batch 100, Loss: 0.7922
实验2 Batch 150, Loss: 0.8426
实验2 Batch 200, Loss: 0.7516
实验2 Batch 250, Loss: 0.9192
实验2 Batch 300, Loss: 0.9744
实验2 Batch 350, Loss: 0.9241
Epoch [8/10], Loss: 0.9268
实验2 Batch 0, Loss: 0.6470
实验2 Batch 50, Loss: 0.5875
实验2 Batch 100, Loss: 0.7008
实验2 Batch 150, Loss: 0.8244
实验2 Batch 200, Loss: 0.8506
实验2 Batch 250, Loss: 1.0092
实验2 Batch 300, Loss: 0.6631
实验2 Batch 350, Loss: 0.4838
Epoch [9/10], Loss: 0.7392
实验2 Batch 0, Loss: 0.7624
实验2 Batch 50, Loss: 0.5368
实验2 Batch 100, Loss: 0.6557
实验2 Batch 150, Loss: 0.6506
实验2 Batch 200, Loss: 0.7720
实验2 Batch 250, Loss: 0.5317
实验2 Batch 300, Loss: 0.4690
实验2 Batch 350, Loss: 0.7022
Epoch [10/10], Loss: 0.6196

--- 实验 3: 换用 Adam 优化器 (基础模型, lr=0.01) ---
实验3 Batch 0, Loss: 2.5197
实验3 Batch 50, Loss: 0.3694
实验3 Batch 100, Loss: 0.2462
实验3 Batch 150, Loss: 0.1626
实验3 Batch 200, Loss: 0.2644
实验3 Batch 250, Loss: 0.8074
实验3 Batch 300, Loss: 0.4653
实验3 Batch 350, Loss: 0.5649
Epoch [1/10], Loss: 0.4073
实验3 Batch 0, Loss: 0.0113
实验3 Batch 50, Loss: 0.0910
实验3 Batch 100, Loss: 0.0563
实验3 Batch 150, Loss: 0.0112
实验3 Batch 200, Loss: 0.0348
实验3 Batch 250, Loss: 0.0289
实验3 Batch 300, Loss: 0.0170
实验3 Batch 350, Loss: 0.1561
Epoch [2/10], Loss: 0.1166
实验3 Batch 0, Loss: 0.0051
实验3 Batch 50, Loss: 0.0058
实验3 Batch 100, Loss: 0.0646
实验3 Batch 150, Loss: 0.0043
实验3 Batch 200, Loss: 0.0027
实验3 Batch 250, Loss: 0.0002
实验3 Batch 300, Loss: 0.0367
实验3 Batch 350, Loss: 0.0559
Epoch [3/10], Loss: 0.0438
实验3 Batch 0, Loss: 0.0083
实验3 Batch 50, Loss: 0.0007
实验3 Batch 100, Loss: 0.0010
实验3 Batch 150, Loss: 0.0003
实验3 Batch 200, Loss: 0.0561
实验3 Batch 250, Loss: 0.0010
实验3 Batch 300, Loss: 0.0009
实验3 Batch 350, Loss: 0.0048
Epoch [4/10], Loss: 0.0183
实验3 Batch 0, Loss: 0.0002
实验3 Batch 50, Loss: 0.0008
实验3 Batch 100, Loss: 0.0045
实验3 Batch 150, Loss: 0.0134
实验3 Batch 200, Loss: 0.0820
实验3 Batch 250, Loss: 0.0184
实验3 Batch 300, Loss: 0.0000
实验3 Batch 350, Loss: 0.0051
Epoch [5/10], Loss: 0.0134
实验3 Batch 0, Loss: 0.0017
实验3 Batch 50, Loss: 0.2887
实验3 Batch 100, Loss: 0.0000
实验3 Batch 150, Loss: 0.1815
实验3 Batch 200, Loss: 0.0027
实验3 Batch 250, Loss: 0.0007
实验3 Batch 300, Loss: 0.0002
实验3 Batch 350, Loss: 0.2140
Epoch [6/10], Loss: 0.0377
实验3 Batch 0, Loss: 0.0003
实验3 Batch 50, Loss: 0.0003
实验3 Batch 100, Loss: 0.0277
实验3 Batch 150, Loss: 0.0045
实验3 Batch 200, Loss: 0.0033
实验3 Batch 250, Loss: 0.0005
实验3 Batch 300, Loss: 0.0005
实验3 Batch 350, Loss: 0.0001
Epoch [7/10], Loss: 0.0372
实验3 Batch 0, Loss: 0.0565
实验3 Batch 50, Loss: 0.0001
实验3 Batch 100, Loss: 0.0242
实验3 Batch 150, Loss: 0.0002
实验3 Batch 200, Loss: 0.0076
实验3 Batch 250, Loss: 0.0002
实验3 Batch 300, Loss: 0.0010
实验3 Batch 350, Loss: 0.0217
Epoch [8/10], Loss: 0.0219
实验3 Batch 0, Loss: 0.0001
实验3 Batch 50, Loss: 0.0003
实验3 Batch 100, Loss: 0.0006
实验3 Batch 150, Loss: 0.0003
实验3 Batch 200, Loss: 0.0019
实验3 Batch 250, Loss: 0.0001
实验3 Batch 300, Loss: 0.0290
实验3 Batch 350, Loss: 0.0004
Epoch [9/10], Loss: 0.0070
实验3 Batch 0, Loss: 0.0000
实验3 Batch 50, Loss: 0.0000
实验3 Batch 100, Loss: 0.0002
实验3 Batch 150, Loss: 0.0001
实验3 Batch 200, Loss: 0.0003
实验3 Batch 250, Loss: 0.0044
实验3 Batch 300, Loss: 0.0000
实验3 Batch 350, Loss: 0.0000
Epoch [10/10], Loss: 0.0050

对比验证完成。

进程已结束，退出代码为 0


"""