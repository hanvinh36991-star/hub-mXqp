import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# BertForSequenceClassification bert 用于 文本分类
# Trainer：直接封装 正向传播、损失计算、参数更新
# TrainingArguments：超参数、实验设置
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# 基础路径：脚本所在目录，避免工作目录不同导致找不到文件
BASE_DIR = Path(__file__).resolve().parent

# 加载并预处理数据（使用当前的 AG News 数据集，只取前200条以加快实验）
dataset_df = pd.read_csv(BASE_DIR / "ag_news_train.csv").head(200)  # 这里替换为现有训练集

# 初始化 LabelEncoder，把文本标签转成数字标签
lbl = LabelEncoder()
labels = lbl.fit_transform(dataset_df["label"].values)  # 使用当前截取的全部样本
# 提取文本内容
texts = list(dataset_df["text"].values)

# 划分训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例20%
    stratify=labels    # 保证训练/测试的标签分布一致
)

# 从本地缓存的预训练模型加载分词器和模型（使用英文 bert-base-uncased 适配 AG News）
# 本机缓存路径：C:\Users\acer\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594
local_model_path = r'C:\Users\acer\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594'
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertForSequenceClassification.from_pretrained(local_model_path, num_labels=len(lbl.classes_))

# 使用分词器对训练集和测试集的文本进行编码
# truncation=True：如果文本过长则截断
# padding=True：对齐所有序列长度，补到最长
# max_length=64：最大序列长度
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# 把编码后的数据和标签转换为 Hugging Face `datasets` 的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID
    'attention_mask': train_encodings['attention_mask'], # attention mask
    'labels': train_labels                               # 对应的标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实标签
    logits, labels = eval_pred
    # 找到 logits 中最大的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算准确率并返回字典
    return {'accuracy': (predictions == labels).mean()}

# 配置训练参数
# 输出目录：按你要求放回 week04（在该目录下建 ASCII 子目录 outputs_agnews）
output_dir = Path(r"D:\NLP\hub-mXqp\张鑫\week04\outputs_agnews")
common_args = dict(
    output_dir=str(output_dir),          # 训练输出目录，用于保存模型和状态
    num_train_epochs=2,                  # 训练轮数
    per_device_train_batch_size=16,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 预热的步数，稳定训练；step 定义为一次 前向传播 + 参数更新
    weight_decay=0.01,                   # L2 正则化
    logging_steps=100,                   # 每隔100步记录一次日志
    save_steps=10,                       # 每10步保存一次 checkpoint
    save_total_limit=2,                  # 只保留最近2个 checkpoint
)

# 兼容旧版 transformers：只传入最少参数，避免 evaluation_strategy/save_strategy/save_safetensors 不被支持
try:
    training_args = TrainingArguments(**common_args)
except TypeError:
    minimal_args = {k: v for k, v in common_args.items() if k not in ["logging_steps"]}
    training_args = TrainingArguments(**minimal_args)

# 实例化 Trainer 封装训练代码
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 验证数据集
    compute_metrics=compute_metrics,     # 用于计算评估指标的函数
)

# 开始训练模型
trainer.train()
# 在测试集上进行最终评估
trainer.evaluate()

# 手动只保存模型权重和分词器到 ASCII 路径
output_dir.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(output_dir))
tokenizer.save_pretrained(str(output_dir))
