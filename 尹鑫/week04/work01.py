import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, \
    recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import chardet

# 读取文件的前几行来检测编码
with open("archive/IMDB Dataset.csv", 'rb') as f:
    result = chardet.detect(f.read())

print(result)

# ==========================
# 数据加载与验证（精简版）
# ==========================
print("正在加载数据...")
dataset_df = pd.read_csv("archive/IMDB Dataset.csv", header=None, encoding='utf-8')

# 基础验证
if dataset_df.isnull().any().any():
    print("⚠️  检测到缺失值，已自动清理")
    dataset_df.dropna(inplace=True)

# 限制样本量（保留原逻辑）
subset_df = dataset_df.head(500) if len(dataset_df) >= 500 else dataset_df
texts = subset_df[0].astype(str).tolist()
labels_raw = subset_df[1].tolist()

# 标签编码（保留编码器用于后续映射）
lbl = LabelEncoder()
labels = lbl.fit_transform(labels_raw)
num_classes = len(np.unique(labels))
print(f"✅ 数据加载完成 | 样本数: {len(texts)} | 类别数: {num_classes}")

# ==========================
# 数据分割
# ==========================
x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)
print(f"📊 训练集: {len(x_train)} | 测试集: {len(x_test)}")

# ==========================
# 模型与分词器
# ==========================
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_classes)

# 编码
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': y_train
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': y_test
})


# ==========================
# 增强版评估指标函数
# ==========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # 计算核心指标
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision,
        'recall_macro': recall
    }


# ==========================
# 训练配置（优化）
# ==========================
os.makedirs('./results', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,  # 评估时增大batch提升速度
    warmup_steps=100,  # 小数据集减少warmup
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",  # 以F1作为最佳模型选择标准
    greater_is_better=True,
    report_to="none"  # 避免wandb等额外日志
)

# ==========================
# 训练
# ==========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("\n🚀 开始训练...")
trainer.train()
print("✅ 训练完成！")

# ==========================
# 【核心增强】全面模型评估
# ==========================
print("\n🔍 正在进行最终评估...")
eval_results = trainer.evaluate()
print("\n📌 测试集评估结果:")
for key, value in eval_results.items():
    if 'loss' not in key:
        print(f"  {key}: {value:.4f}")

# 获取预测结果（用于详细分析）
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

# 1. 详细分类报告（含原始标签名）
print("\n📋 详细分类报告:")
target_names = [str(cls) for cls in lbl.classes_]
report_dict = classification_report(
    true_labels, preds,
    target_names=target_names,
    output_dict=True,
    zero_division=0
)
print(classification_report(true_labels, preds, target_names=target_names, zero_division=0))

# 保存报告到JSON
with open('./results/classification_report.json', 'w', encoding='utf-8') as f:
    json.dump(report_dict, f, ensure_ascii=False, indent=2)
print("✅ 分类报告已保存至: ./results/classification_report.json")

# 2. 混淆矩阵可视化
plt.figure(figsize=(10, 8))
cm = confusion_matrix(true_labels, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('./results/confusion_matrix.png', dpi=150)
print("✅ 混淆矩阵已保存至: ./results/confusion_matrix.png")
plt.close()

# 3. 随机展示5个预测示例（含正确/错误标识）
print("\n🔍 预测示例展示（随机5条）:")
np.random.seed(42)
indices = np.random.choice(len(x_test), min(5, len(x_test)), replace=False)
for i in indices:
    text = x_test[i][:50] + "..." if len(x_test[i]) > 50 else x_test[i]
    true_cls = lbl.inverse_transform([true_labels[i]])[0]
    pred_cls = lbl.inverse_transform([preds[i]])[0]
    status = "✅" if true_labels[i] == preds[i] else "❌"
    print(f"{status} 文本: {text}")
    print(f"   真实标签: {true_cls} | 预测标签: {pred_cls}\n")

# 4. 保存最佳模型（含tokenizer）
print("\n💾 正在保存最佳模型与分词器...")
trainer.save_model('./results/best_model')
tokenizer.save_pretrained('./results/best_model')
print("✅ 模型与分词器已保存至: ./results/best_model")

# 5. 生成评估摘要
summary = {
    "total_samples": len(texts),
    "train_size": len(x_train),
    "test_size": len(x_test),
    "num_classes": num_classes,
    "final_accuracy": float(eval_results['eval_accuracy']),
    "final_f1_macro": float(eval_results['eval_f1_macro']),
    "model_path": "./results/best_model"
}
with open('./results/evaluation_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("✅ 评估摘要已保存至: ./results/evaluation_summary.json")

print("\n🎉 所有评估任务完成！结果已保存至 ./results 目录")
