"""
重新找一个公开数据集， 或 直接标注一个文本分类数据集（推荐3个以上的类别个数），
复现加载bert base 模型在新数据集上的微调过程。最终需要输入一个新的样本进行测试，
验证分类效果是否准确。
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from datasets import Dataset
import numpy as np
import torch
import os

"""
读取数据集：
此数据集用于商品分类
来源：https://www.modelscope.cn/datasets/winwin_inc/product-classification-hiring-demo
"""
BERT_MODEL_PKL_PATH = "./bert.pt"
dataset_df = pd.read_csv("dataset.csv", sep=",", header=0, names=['product_name', 'category'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 前5行数据内容
print(dataset_df.head(5))

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# # 拟合数据并转换前500个标签，得到数字标签
labels = lbl.fit_transform(dataset_df["category"])
# # 提取前500个文本内容
texts = list(dataset_df["product_name"])

num_labels = len(lbl.classes_)
print("类别列表：", lbl.classes_)
print("类别数：", num_labels)

x_train, x_test, train_labels, test_labels = train_test_split(
    texts, # 文本数据
    labels, # 对应的数字标签
    test_size=0.2, # 测试集比例为20%
    stratify=labels  # 确保训练集和测试集的标签分布一致
)

# 从预训练模型加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('D:\\PythonProjects\\models\\google-bert\\bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('D:\\PythonProjects\\models\\google-bert\\bert-base-chinese', num_labels=10)

train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
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

# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': (predictions == labels).mean()}

# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',              # 训练输出目录，用于保存模型和状态
    num_train_epochs=4,                  # 训练的总轮数
    per_device_train_batch_size=16,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练， step 定义为 一次 正向传播 + 参数更新
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=100,                   # 每隔100步记录一次日志
    eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset, # 训练数据集
    eval_dataset = test_dataset, # 测试数据集
    compute_metrics = compute_metrics, # 用于计算评估指标的函数
)

# 检查模型文件是否存在
if os.path.exists(BERT_MODEL_PKL_PATH):
    # 加载现有模型
    model.load_state_dict(torch.load(BERT_MODEL_PKL_PATH))
    print("已加载现有模型")
else:
    # 开始训练模型
    print("未找到模型文件，开始训练...")
    trainer.train()
    trainer.evaluate()

    # 保存训练好的模型
    best_model_path = trainer.state.best_model_checkpoint
    if best_model_path:
        bast_model = BertForSequenceClassification.from_pretrained(best_model_path)
        torch.save(bast_model.state_dict(), './bert.pt')
        print("模型已保存至", BERT_MODEL_PKL_PATH)

# 预测函数, 代码放一起
def predict_category(text):
    test_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    test_inputs = {key: val.to(device) for key, val in test_inputs.items()}  # 移动到模型所在设备

    with torch.no_grad():
        logits = model(**test_inputs).logits

    pred_id = logits.argmax().item()

    # 边界检查并转换标签
    if 0 <= pred_id < len(lbl.classes_):
        pred_label = lbl.inverse_transform([pred_id])[0]
    else:
        pred_label = "未知类别"
    return pred_label


# 最终测试样本预测结果
model.to(device)
model.eval()

input_text = ["椰子水", "甄派火鸡面", "上珍果脆"]
for index, text in enumerate(input_text):
    print("输入样本[{}]：{} => ".format(index, text), end="\t")
    print("预测结果[{}]：{}".format(index, predict_category(text)))
