import pandas as pd

# 构造100条中文文本：50条正面（1）、50条负面（0）
positive_texts = [
    "这部电影太好看了，剧情精彩演员演技也好",
    "今天天气不错，心情特别好",
    "这家餐厅的菜味道很棒，服务也很周到",
    "新买的手机用起来很流畅，性价比超高",
    "和朋友出去玩很开心，下次还想去"
] * 10  # 复制10次，凑50条

negative_texts = [
    "这部电影太烂了，剧情拖沓演员演技差",
    "今天下雨了，出门很不方便，心情很差",
    "这家餐厅的菜很难吃，服务态度也不好",
    "新买的手机经常卡顿，性价比超低",
    "上班迟到了，被老板批评，心情糟糕"
] * 10  # 复制10次，凑50条

# 组合文本和标签（修改点1：标签长度匹配文本，50正+50负）
all_texts = positive_texts + negative_texts  # 长度=100
all_labels = [1]*50 + [0]*50  # 长度=100（原代码是20+20，改为50+50）

# 保存为Tab分隔的CSV文件（无表头）（修改点2：sep="\t"，和读取时一致）
test_dataset = pd.DataFrame({0: all_texts, 1: all_labels})
test_dataset.to_csv("dataset.csv", sep="\t", header=None, index=False)

print("测试数据集已生成：dataset.csv")

# ========== 第二部分：模型训练与评估 ==========
import jieba
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- 1. 数据加载与预处理 ---
print("\n--- 1. 正在加载和预处理数据 ---")
# 修改点3：文件路径改为当前目录的dataset.csv（原代码是../Week01/dataset.csv）
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=100)

print("数据集前5行预览：")
print(dataset.head(5))

# 使用jieba进行中文分词，并用空格分隔
input_sentences = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
labels = dataset[1]

# --- 2. 特征提取 (词袋模型) ---
print("\n--- 2. 正在进行特征提取 ---")
vectorizer = CountVectorizer()
input_features = vectorizer.fit_transform(input_sentences.values)
print(f"特征矩阵的形状： {input_features.shape}")

# --- 3. 模型训练与评估 ---
models = {
    'KNN (K-Nearest Neighbors)': KNeighborsClassifier(),
    'Naive Bayes (MultinomialNB)': MultinomialNB(),
    'SVM (Support Vector Classifier)': SVC(kernel='linear'),
    'Decision Tree': DecisionTreeClassifier()
}

for name, model in models.items():
    print(f"\n--- 正在评估模型: {name} ---")
    y_pred = cross_val_predict(model, input_features, labels, cv=5)
    print(f"平均准确率: {accuracy_score(labels, y_pred):.4f}")
    report = classification_report(labels, y_pred, zero_division=0)
    print(report)
