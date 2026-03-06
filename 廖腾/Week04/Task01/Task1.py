import pandas as pd
import torch
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

# 设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# 1. 加载数据
print("加载数据...")

def load_cnews_file(file_path):
    """加载cnews文件，格式：标签\t内容"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                label, text = line.split('\t', 1)
                data.append({'label': label.strip(), 'text': text.strip()})
    return pd.DataFrame(data)

# 加载三个文件
train_path = r'E:\YunZjDownload\pytorch-learning\Ai_study\Week04_Dataset\cnews\cnews.train.sample.txt'
val_path = r'E:\YunZjDownload\pytorch-learning\Ai_study\Week04_Dataset\cnews\cnews.val.sample.txt'
test_path = r'E:\YunZjDownload\pytorch-learning\Ai_study\Week04_Dataset\cnews\cnews.test.sample.txt'

# 加载数据
train_df = load_cnews_file(train_path)
val_df = load_cnews_file(val_path)
test_df = load_cnews_file(test_path)

# 抽样（如果你只需要部分数据）
sample_size = 500  # 每个文件抽取500条
if sample_size:
    train_df = train_df.sample(min(sample_size, len(train_df)), random_state=42)
    val_df = val_df.sample(min(sample_size, len(val_df)), random_state=42)
    test_df = test_df.sample(min(sample_size, len(test_df)), random_state=42)

# 提取文本和标签
x_train = train_df['text'].tolist()
train_labels = train_df['label'].tolist()

x_val = val_df['text'].tolist()
val_labels = val_df['label'].tolist()

x_test = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

print(f"训练集: {len(x_train)}条")
print(f"验证集: {len(x_val)}条")
print(f"测试集: {len(x_test)}条")

# 2. 标签编码
label_encoder = LabelEncoder()
all_labels = train_labels + val_labels + test_labels
label_encoder.fit(all_labels)

train_labels_encoded = label_encoder.transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
test_labels_encoded = label_encoder.transform(test_labels)

print(f"类别数: {len(label_encoder.classes_)}")

# 3. 加载BERT模型
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=len(label_encoder.classes_),
    ignore_mismatched_sizes=True
)

# 4. 数据预处理
def tokenize_function(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors=None
    )

# 创建Dataset
train_dataset = Dataset.from_dict({
    'text': x_train,
    'labels': train_labels_encoded
})
val_dataset = Dataset.from_dict({
    'text': x_val,
    'labels': val_labels_encoded
})
test_dataset = Dataset.from_dict({
    'text': x_test,
    'labels': test_labels_encoded
})

# 应用分词
train_dataset = train_dataset.map(lambda x: tokenize_function(x['text']), batched=True)
val_dataset = val_dataset.map(lambda x: tokenize_function(x['text']), batched=True)
test_dataset = test_dataset.map(lambda x: tokenize_function(x['text']), batched=True)

# 设置格式
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 5. 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

# 6. 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
)

# 7. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# 8. 训练
print("开始训练...")
trainer.train()

# 9. 评估
print("评估模型...")
eval_results = trainer.evaluate()
print(f"验证集准确率: {eval_results['eval_accuracy']:.4f}")

# 10. 保存模型
trainer.save_model("./cnews_bert_model")

print("训练完成!")