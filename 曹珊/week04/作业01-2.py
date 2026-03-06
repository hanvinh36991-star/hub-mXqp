"""
1. 重新找一个公开数据集，或直接标注一个文本分类数据集（推荐3个以上的类别个数），完成bert的微调过程
"""

# -------------------------- 1. 数据准备 --------------------------
# 加载数据集，指定分隔符为制表符，并无表头
import json

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("./chinese-logic-sentiment-dataset.csv", sep=",", header=None)
print(dataset.head(5))
text_have_head = dataset[0].tolist()
label_have_head = dataset[4].tolist()
texts = text_have_head[1:]
labels = label_have_head[1:]

# 初始化并拟合标签编码器，将文本标签（如“体育”）转换为数字标签（如0, 1, 2...）
lbl = LabelEncoder()
labels = lbl.fit_transform(labels)

# 保存标签映射，用于后续中文输出
label_mapping = {i: label for i, label in enumerate(lbl.classes_)}
print(f"\n标签映射关系 (数字 -> 中文):")
for num, text in label_mapping.items():
    print(f"  {num}: {text}")

with open('label_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=2)
print("✅ 标签映射已保存到 label_mapping.json")

# 将数据按8:2的比例分割为训练集和测试集
# stratify 参数确保训练集和测试集中各类别的样本比例与原始数据集保持一致
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels
)

# 从预训练模型加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('./models/google-bert/bert-base-chinese', num_labels=6)

# 使用分词器对训练集和测试集的文本进行编码
# truncation=True：如果文本过长则截断
# padding=True：对齐所有序列长度，填充到最长
# max_length=64：最大序列长度
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# -------------------------- 2. 数据集和数据加载器 --------------------------
# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],  # 文本的token ID
    'attention_mask': train_encodings['attention_mask'],  # 注意力掩码
    'labels': train_labels  # 对应的标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# -------------------------- 3. 模型和优化器 --------------------------
# 加载BERT用于序列分类的预训练模型
# num_labels=12：指定分类任务的类别数量
# https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/bert#transformers.BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('./models/google-bert/bert-base-chinese', num_labels=6,
                                                      ignore_mismatched_sizes=True)

# 设置设备，优先使用CUDA（GPU），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移动到指定的设备上
model.to(device)

# 定义优化器，使用AdamW，lr是学习率
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)


# 定义精度计算函数
def flat_accuracy(preds, labels):
    # 获取预测结果的最高概率索引
    pred_flat = np.argmax(preds, axis=1).flatten()
    # 展平真实标签
    labels_flat = labels.flatten()
    # 计算准确率
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}


# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',  # 训练输出目录，用于保存模型和状态
    num_train_epochs=4,  # 训练的总轮数
    per_device_train_batch_size=16,  # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,  # 评估时每个设备的批次大小
    warmup_steps=500,  # 学习率预热的步数，有助于稳定训练， step 定义为 一次 正向传播 + 参数更新
    weight_decay=0.01,  # 权重衰减，用于防止过拟合
    logging_dir='./logs',  # 日志存储目录
    logging_steps=100,  # 每隔100步记录一次日志
    eval_strategy="epoch",  # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",  # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,  # 训练结束后加载效果最好的模型
)

# 实例化 Trainer 简化模型训练代码
trainer = Trainer(
    model=model,  # 要训练的模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=test_dataset,  # 评估数据集
    compute_metrics=compute_metrics,  # 用于计算评估指标的函数
)

# 深度学习训练过程，数据获取，epoch batch 循环，梯度计算 + 参数更新

# 开始训练模型
trainer.train()
# 在测试集上进行最终评估
trainer.evaluate()


best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    best_model = BertForSequenceClassification.from_pretrained(best_model_path)
    print(f"The best model is located at: {best_model_path}")
    torch.save(best_model.state_dict(), './results/bert.pt')
    print("Best model saved to assets/weights/bert.pt")
else:
    print("Could not find the best model checkpoint.")
