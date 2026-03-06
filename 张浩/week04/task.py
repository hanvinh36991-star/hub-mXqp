import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 配置参数
BERT_MODEL_PRETRAINED_PATH = "bert-base-chinese"  
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_and_prepare_data(file_path):

    df = pd.read_csv(file_path)

    # 获取唯一类别并创建标签映射
    unique_categories = df['category'].unique()
    category_to_id = {category: idx for idx, category in enumerate(unique_categories)}
    id_to_category = {idx: category for category, idx in category_to_id.items()}

    print(f"发现 {len(unique_categories)} 个类别: {list(unique_categories)}")

    # 将类别转换为数字标签
    texts = df['text'].tolist()
    labels = [category_to_id[cat] for cat in df['category']]

    return texts, labels, category_to_id, id_to_category


def train_model(model, train_dataloader, val_dataloader, epochs=EPOCHS):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)

        # 验证模型
        model.eval()
        val_predictions = []
        val_labels = []
        for batch in val_dataloader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_predictions)
      
        model.train()

    return model


def predict_texts(model, tokenizer, texts, id_to_category, max_length=MAX_LENGTH):
    model.eval()
    predictions = []

    for text in texts:
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
            predictions.append(id_to_category[predicted_class])

    return predictions


def main():
    # 加载数据
    print("正在加载数据集...")
    texts, labels, category_to_id, id_to_category = load_and_prepare_data('news_dataset.csv')

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")

    # 初始化tokenizer和模型
    print("初始化BERT模型...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PRETRAINED_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_PRETRAINED_PATH,
        num_labels=len(category_to_id)
    )
    model.to(device)

    # 创建数据集和数据加载器
    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    val_dataset = NewsDataset(X_val, y_val, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 训练模型
    print("开始训练模型...")
    trained_model = train_model(model, train_dataloader, val_dataloader, epochs=EPOCHS)

    # 测试新样本
    print("\n训练完成！现在进行预测测试...")

    # 测试新的样本（选择一些不在原始数据中的示例）
    test_texts = [
        "人工智能在医疗诊断中取得突破性进展",
        "NBA总决赛激烈对决，最终冠军诞生",
        "最新智能手机发布，搭载先进芯片技术",
        "股市今日大幅波动，投资者需谨慎操作"
    ]

    predictions = predict_texts(trained_model, tokenizer, test_texts, id_to_category)

    print("\n预测结果:")
    for text, pred in zip(test_texts, predictions):
        print(f"文本: '{text}' -> 预测类别: {pred}")

    print("\n模型微调和预测流程已完成！")


if __name__ == "__main__":
    main()
