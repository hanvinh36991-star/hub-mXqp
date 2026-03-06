import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, classification_report

# 1. 配置基础参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-uncased"
batch_size = 16
epochs = 5  # 多类别需更多训练轮数
learning_rate = 1e-5  # 微调学习率略低

# 2. 加载CLINC150数据集
dataset = load_dataset("clinc_oos", "plus")  # plus版本含150类+OOS

# 构建标签映射
unique_labels = sorted(list(set(dataset["train"]["intent"])))
id2label = {i: label for i, label in enumerate(unique_labels)}
label2id = {label: i for i, label in enumerate(unique_labels)}
num_labels = len(id2label)

# 重新映射标签为数字
def remap_label(example):
    example["label"] = label2id[example["intent"]]
    example["text"] = example["text"]
    return example

dataset = dataset.map(remap_label)

# 3. 分词预处理
tokenizer = BertTokenizer.from_pretrained(model_name)
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,  # 意图文本短，64足够
        padding="max_length"
    )

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# 4. 加载模型
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device)

# 5. 评估指标（适配多类别）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # 只打印关键指标，避免150类报告过长
    return {
        "accuracy": accuracy_score(labels, predictions),
        "top3_accuracy": np.mean([1 if l in p else 0 for l, p in 
                                 zip(labels, np.argsort(logits, axis=-1)[:, -3:])])
    }

# 6. 训练配置（适配多类别）
training_args = TrainingArguments(
    output_dir="./bert_clinc150",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs_clinc150",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    warmup_ratio=0.1,  # 预热学习率，适配多类别
)

# 7. 训练与评估
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train()
eval_results = trainer.evaluate(encoded_dataset["test"])
print("\n=== CLINC150 测试集评估结果 ===")
print(f"准确率: {eval_results['eval_accuracy']:.4f}")
print(f"Top3准确率: {eval_results['eval_top3_accuracy']:.4f}")

# 8. 新样本预测
def predict_text(text):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=64,
        padding="max_length",
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    pred_label_id = torch.argmax(logits, dim=-1).item()
    pred_label = id2label[pred_label_id]
    confidence = torch.softmax(logits, dim=-1)[0][pred_label_id].item()
    
    # 输出Top3预测结果（多类别场景更实用）
    top3_ids = torch.argsort(logits, dim=-1)[0][-3:].cpu().numpy()[::-1]
    top3_labels = [id2label[id] for id in top3_ids]
    top3_confidences = [torch.softmax(logits, dim=-1)[0][id].item() for id in top3_ids]
    
    return {
        "文本": text,
        "预测类别(Top1)": pred_label,
        "置信度(Top1)": f"{confidence:.4f}",
        "Top3预测": list(zip(top3_labels, [f"{c:.4f}" for c in top3_confidences]))
    }

# 测试意图样本
print("\n=== 新样本测试 ===")
test_samples = [
    "What's the weather like in New York tomorrow?",  # weather
    "Can I transfer $500 to my savings account?",     # transfer_money
    "Book a flight from LA to Chicago next Monday",   # book_flight
    "How do I reset my password?",                    # reset_password
    "What's the capital of France?"                   # OOS（域外意图）
]

for sample in test_samples:
    result = predict_text(sample)
    print(f"{result}\n")
