import os
# 设置 Hugging Face 镜像源，解决连接超时问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# 1. 加载数据
print("正在加载数据...")
# 使用逗号分隔
dataset_df = pd.read_csv("dataset_v2.csv", sep=",") 

# 2. 标签编码
print("正在处理标签...")
lbl = LabelEncoder()
dataset_df['label_id'] = lbl.fit_transform(dataset_df['label'])
labels = dataset_df['label_id'].values
texts = dataset_df['text'].values

# 打印一下类别映射关系，方便查看
label_map = {index: label for index, label in enumerate(lbl.classes_)}
print(f"类别映射: {label_map}")

# 3. 分割数据集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts, 
    labels, 
    test_size=0.2, 
    stratify=labels,
    random_state=42
)

# 4. 加载模型和分词器
model_name = 'bert-base-chinese'
print(f"正在加载模型: {model_name} ...")
try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))
except Exception as e:
    print(f"模型加载失败，请检查网络或路径: {e}")
    exit(1)

# 5. 数据预处理（Tokenization）
def tokenize_function(texts):
    return tokenizer(list(texts), truncation=True, padding=True, max_length=64)

train_encodings = tokenize_function(x_train)
test_encodings = tokenize_function(x_test)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# 6. 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': (predictions == labels).mean()}

# 7. 训练参数
training_args = TrainingArguments(
    output_dir='./results_task',
    num_train_epochs=3,              # 演示用，3轮应该很快（数据量小）
    per_device_train_batch_size=4,   # 数据量少，batch size 小一点
    per_device_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs_task',
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 8. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("开始训练...")
trainer.train()
print("训练完成。")

print("在测试集上评估...")
eval_result = trainer.evaluate()
print(f"评估结果: {eval_result}")

# 9. 预测新样本
def predict_sentence(sentence):
    print(f"\n正在预测句子: '{sentence}'")
    # 预处理
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=64)
    # 将输入移动到模型所在的设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        
    predicted_label = label_map[predicted_class_id]
    print(f"预测结果: {predicted_label} (ID: {predicted_class_id})")
    return predicted_label

# 测试几个新句子
test_sentences = [
    "英伟达发布了最新的显卡",    # Technology
    "昨晚的球赛太精彩了",        # Sports
    "一定要注意理财风险",        # Finance
    "AI将取代很多工作"           # Technology
]

print("\n=== 新样本测试 ===")
for s in test_sentences:
    predict_sentence(s)
