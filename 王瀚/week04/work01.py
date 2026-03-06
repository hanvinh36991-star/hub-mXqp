import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warming
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from datasets import load_dataset
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子保证可复现
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 1. 数据准备 ====================
print("\n=== 1. 加载并改造数据集 ===")

# 加载IMDb数据集（原本是二分类：pos/neg）
dataset = load_dataset("imdb")
train_data = dataset['train']
test_data = dataset['test']


# 改造成3分类：通过评分阈值划分
# 原数据：1-4分(neg), 9-10分(pos) -> 我们改造成：1-3(消极), 4-7(中性), 8-10(积极)
def convert_to_3class(text, label):
    """
    将二分类改造成3分类：
    - 原label=0(负面): 拆分为强负面(0)和中性偏负(1)
    - 原label=1(正面): 拆分为强正面(2)和中性偏正(1)
    基于文本长度简单模拟：短文本为极端情绪，长文本为温和情绪
    """
    text_length = len(text.split())
    if label == 0:  # 原本是负面
        return 0 if text_length > 200 else 1  # 长=强负面，短=中性偏负
    else:  # 原本是正面
        return 2 if text_length > 200 else 1  # 长=强正面，短=中性偏正


# 构建3分类数据集
def build_3class_dataset(split_data, max_samples=3000):  # 限制样本数以加速训练
    texts = []
    labels = []

    for i, item in enumerate(split_data):
        if i >= max_samples:
            break
        text = item['text']
        orig_label = item['label']
        new_label = convert_to_3class(text, orig_label)
        texts.append(text)
        labels.append(new_label)

    return pd.DataFrame({'text': texts, 'label': labels})


print("构建3分类数据集...")
train_df = build_3class_dataset(train_data, max_samples=3000)
test_df = build_3class_dataset(test_data, max_samples=500)

print(f"训练集分布:\n{train_df['label'].value_counts().sort_index()}")
print(f"测试集分布:\n{test_df['label'].value_counts().sort_index()}")
print("类别映射: 0=消极, 1=中性, 2=积极")

# ==================== 2. 数据集类定义 ====================
print("\n=== 2. 定义PyTorch Dataset ===")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ==================== 3. 初始化BERT ====================
print("\n=== 3. 加载BERT-base-uncased ===")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3,  # 3分类
    output_attentions=False,
    output_hidden_states=False
)
model = model.to(device)

# 查看模型结构
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# ==================== 4. 创建DataLoader ====================
print("\n=== 4. 准备数据加载器 ===")

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].values,
    train_df['label'].values,
    test_size=0.1,
    random_state=42,
    stratify=train_df['label']
)

# 创建datasets
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
test_dataset = TextDataset(test_df['text'].values, test_df['label'].values, tokenizer)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}, 测试样本: {len(test_dataset)}")

# ==================== 5. 训练配置 ====================
print("\n=== 5. 配置训练参数 ===")

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
epochs = 3
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warming(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss()

# ==================== 6. 训练循环 ====================
print("\n=== 6. 开始微调训练 ===")


def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(data_loader)


def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))

    return avg_loss, accuracy, predictions, true_labels


# 训练循环
best_accuracy = 0
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_acc, _, _ = eval_model(model, val_loader, device)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), 'best_bert_3class.pt')
        print("✓ 保存最佳模型")

# 加载最佳模型
model.load_state_dict(torch.load('best_bert_3class.pt'))

# ==================== 7. 测试集评估 ====================
print("\n=== 7. 在测试集上评估 ===")

test_loss, test_acc, predictions, true_labels = eval_model(model, test_loader, device)
print(f"\n测试集准确率: {test_acc:.4f}")

print("\n分类报告:")
target_names = ['消极(Negative)', '中性(Neutral)', '积极(Positive)']
print(classification_report(true_labels, predictions, target_names=target_names))

print("混淆矩阵:")
print(confusion_matrix(true_labels, predictions))

# ==================== 8. 新样本测试 ====================
print("\n=== 8. 新样本测试验证 ===")


def predict_text(text, model, tokenizer, device):
    """预测单条文本的情感类别"""
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    label_map = {0: '消极', 1: '中性', 2: '积极'}
    return label_map[pred_label], confidence, probs[0].cpu().numpy()


# 构造3个新样本，分别对应3个类别
test_samples = [
    {
        "text": "This movie is absolutely terrible. The plot makes no sense, acting is horrible, and I wasted two hours of my life. Complete disaster and worst film I've ever seen!",
        "expected": "消极",
        "desc": "强负面评价"
    },
    {
        "text": "The movie was okay. Some parts were good, some were boring. Not the best I've seen but not the worst either. Might watch it again if there's nothing else on.",
        "expected": "中性",
        "desc": "中性评价"
    },
    {
        "text": "Absolutely fantastic! Best movie of the year! The cinematography, acting, and story were all perfect. I was moved to tears and will definitely watch it again!",
        "expected": "积极",
        "desc": "强正面评价"
    }
]

print("\n对新样本进行分类测试:\n")
for i, sample in enumerate(test_samples, 1):
    pred_label, confidence, all_probs = predict_text(sample['text'], model, tokenizer, device)

    print(f"样本 {i}: {sample['desc']}")
    print(f"文本: {sample['text'][:100]}...")
    print(f"期望类别: {sample['expected']}")
    print(f"预测类别: {pred_label} (置信度: {confidence:.4f})")
    print(f"各类别概率: 消极={all_probs[0]:.4f}, 中性={all_probs[1]:.4f}, 积极={all_probs[2]:.4f}")

    # 验证是否正确
    is_correct = (pred_label == sample['expected'])
    print(f"结果: {'✓ 正确' if is_correct else '✗ 错误'}")
    print("-" * 80)

print("\n=== 完成！BERT-base微调已成功应用于3分类任务 ===")
