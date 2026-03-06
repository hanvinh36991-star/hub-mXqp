import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')
model = AutoModelForSequenceClassification.from_pretrained('./models/google-bert/bert-base-chinese', num_labels=6)

model.load_state_dict(torch.load("./results/bert.pt"))

# 设置为评估模式
model.eval()

# 如果有GPU，可以移到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def prodict(text: str):
    # 编码文本
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )

    # 移动到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)

    # 获取标签值
    predicted_label_idx = torch.argmax(probabilities, dim=-1).item()

    with open('label_mapping.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("所有的标签",data)
    return data[str(predicted_label_idx)]


if __name__ == '__main__':
    text1 = "买的这个抽真空机真好用，又便宜，以后冰箱再也不会这么乱了"
    print("打印原文",text1)
    print("打印预测结果", prodict(text1))

    text2 = "过年回来，要不要一起打球"
    print("打印预测结果", prodict(text2))

    text3 = "这个太好吃了，我非常喜欢"
    print("打印预测结果", prodict(text3))

    text4 = "前两天打雷下雨，把小猫咪吓坏了"
    print("打印预测结果", prodict(text4))

    text5 = "吉隆坡好不好玩？"
    print("打印预测结果", prodict(text5))
