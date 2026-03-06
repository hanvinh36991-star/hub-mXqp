import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("customer_shopping_behavior.csv")


# 创建文本特征：结合多个字段生成描述性文本
def create_text_features(row):
    text_parts = [
        f"客户年龄: {row['age']}岁",
        f"性别: {row['gender']}",
        f"年收入: ${row['annual_income']:,.0f}",
        f"消费分数: {row['spending_score']}",
        f"购买频率: 每月{row['purchase_frequency']}次",
        f"平均交易额: ${row['avg_transaction_value']:.2f}",
        f"去年总消费: ${row['total_spent_last_year']:.2f}",
        f"会员时长: {row['membership_months']}个月",
        f"在线购物比例: {row['online_shopping_ratio'] * 100:.0f}%",
        f"客户价值分数: {row['customer_value_score']:.2f}"
    ]
    return "，".join(text_parts)


# 应用函数创建文本
df['text'] = df.apply(create_text_features, axis=1)

# 选择标签（customer_category有多个类别）
labels = df['customer_category'].values
texts = df['text'].values

print(f"总样本数: {len(texts)}")
print(f"文本示例: {texts[0][:150]}...")
print(f"对应标签: {labels[0]}")

#  编码标签
lbl = LabelEncoder()
encoded_labels = lbl.fit_transform(labels)
num_classes = len(lbl.classes_)

print(f"类别数量: {num_classes}")
print(f"类别映射: {dict(zip(lbl.classes_, range(num_classes)))}")
print(f"标签分布:\n{pd.Series(encoded_labels).value_counts().sort_index()}")

# 4. 分割数据集
x_train, x_test, y_train, y_test = train_test_split(
    texts, encoded_labels,
    test_size=0.2,
    stratify=encoded_labels,
    random_state=42
)

print(f"训练集大小: {len(x_train)}")
print(f"测试集大小: {len(x_test)}")
print(f"训练集标签分布:\n{pd.Series(y_train).value_counts().sort_index()}")

# 5. 加载BERT模型和分词器
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=num_classes
    )
    print("成功加载BERT中文模型")
except Exception as e:
    print(f"加载模型时出错: {e}")
    print("尝试从HuggingFace下载...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=num_classes
    )

#  编码文本数据


def tokenize_function(texts):
    return tokenizer(
        list(texts),
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )


train_encodings = tokenize_function(x_train)
test_encodings = tokenize_function(x_test)

# 创建PyTorch数据集

class CustomerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = CustomerDataset(train_encodings, y_train)
test_dataset = CustomerDataset(test_encodings, y_test)

print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

#  训练参数设置
training_args = TrainingArguments(
    output_dir='./customer_classification_results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./customer_logs',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    report_to="none",
    learning_rate=2e-5,
    gradient_accumulation_steps=2,
)

# 创建Trainer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 开始训练
train_result = trainer.train()
print(f"训练完成！训练损失: {train_result.training_loss:.4f}")

# 评估模型
eval_result = trainer.evaluate()
print(f"评估结果:")
for key, value in eval_result.items():
    print(f"  {key}: {value:.4f}")

# 保存模型
trainer.save_model("./customer_classification_model")
tokenizer.save_pretrained("./customer_classification_model")
print("模型已保存到 './customer_classification_model'")

# 验证文本分类

def predict_customer_category(text):
    """预测客户类别"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        predicted_prob = probabilities[0][predicted_class].item()

    predicted_label = lbl.inverse_transform([predicted_class])[0]

    return {
        'predicted_category': predicted_label,
        'confidence': predicted_prob,
        'class_probabilities': probabilities.numpy()[0]
    }


# 从测试集中选择几个样本进行验证
test_indices = np.random.choice(len(x_test), min(5, len(x_test)), replace=False)

print("\n测试样本预测结果:")
print("=" * 100)

for i, idx in enumerate(test_indices):
    test_text = x_test[idx]
    true_label = lbl.inverse_transform([y_test[idx]])[0]

    # 简化显示文本
    short_text = test_text[:100] + "..." if len(test_text) > 100 else test_text

    # 进行预测
    result = predict_customer_category(test_text)

    print(f"\n样本 {i + 1}:")
    print(f"客户特征: {short_text}")
    print(f"真实类别: {true_label}")
    print(f"预测类别: {result['predicted_category']}")
    print(f"置信度: {result['confidence']:.2%}")

    # 显示所有类别的概率
    print("类别概率分布:")
    for class_idx, class_name in enumerate(lbl.classes_):
        prob = result['class_probabilities'][class_idx]
        if prob > 0.05:  # 只显示概率大于5%的类别
            print(f"  {class_name}: {prob:.2%}")

    # 判断是否正确
    if result['predicted_category'] == true_label:
        print("✓ 预测正确！")
    else:
        print("✗ 预测错误！")

    print("-" * 80)

# 完整测试集评估

def evaluate_on_test_set(dataset, model, tokenizer, label_encoder):
    """在完整测试集上进行评估"""
    model.eval()
    all_predictions = []
    all_labels = []

    for i in range(len(dataset)):
        # 获取数据
        item = dataset[i]
        input_ids = item['input_ids'].unsqueeze(0)
        attention_mask = item['attention_mask'].unsqueeze(0)
        label = item['labels'].item()

        # 预测
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()

        all_predictions.append(prediction)
        all_labels.append(label)

    # 计算准确率
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    # 计算每个类别的准确率
    from sklearn.metrics import classification_report, confusion_matrix

    predictions_decoded = label_encoder.inverse_transform(all_predictions)
    labels_decoded = label_encoder.inverse_transform(all_labels)

    print(f"总体准确率: {accuracy:.2%}")
    print("\n分类报告:")
    print(classification_report(labels_decoded, predictions_decoded, target_names=label_encoder.classes_))

    # 创建混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    return accuracy, cm


# 执行完整评估
accuracy, cm = evaluate_on_test_set(test_dataset, model, tokenizer, lbl)

#  可视化结果

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=lbl.classes_,
            yticklabels=lbl.classes_)
plt.title('客户类别分类混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("混淆矩阵已保存为 'confusion_matrix.png'")

# 创建预测函数示例


def predict_from_customer_data(customer_data):
    """
    根据客户数据预测类别
    customer_data: 字典，包含以下字段（与CSV列对应）:
        age, gender, annual_income, spending_score,
        purchase_frequency, avg_transaction_value,
        total_spent_last_year, membership_months,
        online_shopping_ratio, customer_value_score
    """
    # 创建文本
    text = create_text_features(customer_data)

    # 预测
    result = predict_customer_category(text)

    print(f"客户特征: {text[:150]}...")
    print(f"预测客户类别: {result['predicted_category']}")
    print(f"置信度: {result['confidence']:.2%}")

    return result


# 测试示例
print("\n示例预测:")
example_customer = {
    'age': 30,
    'gender': 'Female',
    'annual_income': 50000,
    'spending_score': 75,
    'purchase_frequency': 12,
    'avg_transaction_value': 200.0,
    'total_spent_last_year': 2400.0,
    'membership_months': 24,
    'online_shopping_ratio': 0.6,
    'customer_value_score': 2500.0
}

result = predict_from_customer_data(example_customer)

print("\n" + "=" * 100)
print("BERT微调总结:")
print(f"1. 数据集: {df.shape[0]} 个客户样本")
print(f"2. 分类任务: {num_classes} 个客户类别")
print(f"3. 训练准确率: {train_result.metrics['train_loss']:.4f} (损失)")
print(f"4. 测试准确率: {accuracy:.2%}")
print(f"5. 模型已保存到: ./customer_classification_model/")
print(f"6. 类别包括: {', '.join(lbl.classes_)}")
print("=" * 100)

print("\n BERT文本分类微调完成！模型可以用于预测客户类别。")
