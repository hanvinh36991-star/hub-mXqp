# bert_cpu_only.py
import os

# 在导入任何torch相关库之前设置环境变量
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'  # 完全禁用MPS
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用CUDA

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
	BertTokenizer,
	BertForSequenceClassification,
	Trainer,
	TrainingArguments
)
from datasets import Dataset
import warnings

warnings.filterwarnings('ignore')

# 强制使用CPU
torch.device('cpu')

print("BERT文本分类 - CPU专用版本")
print("=" * 60)
print(f"PyTorch设备: {torch.device('cpu')}")
print(f"MPS可用: {torch.backends.mps.is_available()}")
print(f"CUDA可用: {torch.cuda.is_available()}")


# 1. 创建数据集
def create_simple_dataset():
	categories = ["体育", "科技", "财经", "娱乐", "教育"]
	texts = []
	labels = []

	# 每个类别生成数据
	for label_idx, category in enumerate(categories):
		for i in range(40):  # 每个类别40个样本
			if category == "体育":
				text = f"篮球比赛第{i}场非常精彩，双方队员表现出色"
			elif category == "科技":
				text = f"人工智能技术取得新突破{i}，改变行业发展"
			elif category == "财经":
				text = f"股市今日上涨{i}个百分点，投资者信心增强"
			elif category == "娱乐":
				text = f"最新电影票房创新高{i}，观众评价很好"
			else:  # 教育
				text = f"学校推出新课程{i}，培养学生综合能力"

			texts.append(text)
			labels.append(label_idx)

	return texts, labels, categories


print("1. 创建数据集...")
texts, labels, categories = create_simple_dataset()
print(f"   样本数: {len(texts)}")
print(f"   类别: {categories}")

# 2. 分割数据
train_texts, test_texts, train_labels, test_labels = train_test_split(
	texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"   训练集: {len(train_texts)}")
print(f"   测试集: {len(test_texts)}")

# 3. 创建datasets
train_dataset = Dataset.from_dict({
	'text': train_texts,
	'label': train_labels
})

test_dataset = Dataset.from_dict({
	'text': test_texts,
	'label': test_labels
})

# 4. 加载tokenizer和模型
print("\n2. 加载模型...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 强制在CPU上加载模型
with torch.device('cpu'):
	model = BertForSequenceClassification.from_pretrained(
		'bert-base-chinese',
		num_labels=len(categories)
	)


# 5. 分词函数
def tokenize_function(examples):
	return tokenizer(
		examples['text'],
		truncation=True,
		padding=True,
		max_length=64,  # 减小长度节省内存
		return_tensors=None
	)


print("3. 分词处理...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 移除文本列
train_dataset = train_dataset.remove_columns(['text'])
test_dataset = test_dataset.remove_columns(['text'])

# 6. 设置训练参数（强制使用CPU）
print("4. 设置训练参数...")
training_args = TrainingArguments(
	output_dir='./bert_cpu_results',
	num_train_epochs=2,  # 减少轮数
	per_device_train_batch_size=4,  # 减小batch size
	per_device_eval_batch_size=4,
	learning_rate=2e-5,
	weight_decay=0.01,
	logging_steps=5,
	eval_strategy="no",
	save_strategy="no",
	report_to="none",
	no_cuda=True,  # 明确禁用CUDA
	use_cpu=True,  # 明确使用CPU
)


# 7. 评估函数
def compute_metrics(p):
	predictions, labels = p
	predictions = np.argmax(predictions, axis=1)
	return {"accuracy": accuracy_score(labels, predictions)}


# 8. 创建Trainer
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	compute_metrics=compute_metrics,
)

# 9. 训练模型
print("\n5. 开始训练...")
trainer.train()

# 10. 评估模型
print("\n6. 评估模型...")
results = trainer.evaluate()
print(f"   测试准确率: {results['eval_accuracy']:.4f}")


# 11. 安全预测函数（强制CPU）
def safe_predict(text, model, tokenizer, categories):
	"""完全在CPU上运行的预测函数"""
	# 确保模型在CPU上
	model = model.to('cpu')
	model.eval()

	# 编码文本
	encoding = tokenizer(
		text,
		return_tensors='pt',
		truncation=True,
		max_length=64
	)

	# 确保输入在CPU上
	encoding = {k: v.to('cpu') for k, v in encoding.items()}

	with torch.no_grad():
		outputs = model(**encoding)
		predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
		predicted_idx = torch.argmax(predictions, dim=1).item()
		confidence = predictions[0][predicted_idx].item()

	return categories[predicted_idx], confidence


# 12. 测试预测
print("\n" + "=" * 60)
print("测试预测")
print("=" * 60)

test_samples = [
	"篮球比赛非常精彩，观众热情高涨",
	"人工智能技术快速发展，应用广泛",
	"股市行情看好，投资者积极入场",
	"电影票房大卖，口碑很好",
	"学校教育改革，课程创新"
]

for i, text in enumerate(test_samples, 1):
	predicted_label, confidence = safe_predict(text, model, tokenizer, categories)
	print(f"{i}. '{text[:20]}...'")
	print(f"   预测: {predicted_label} (置信度: {confidence:.4f})")
	print()

print("✅ 训练完成！")
print(f"最终准确率: {results['eval_accuracy']:.4f}")