import os
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

# ================= 1. 环境配置 =================

# 初始化 OpenAI 客户端 (阿里云 DashScope)
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义分类标签池，确保 LLM 遵循一致的输出
CLASSIFICATION_LABELS = """
FilmTele-Play, Video-Play, Music-Play, Radio-Listen, Alarm-Update, 
Travel-Query, HomeAppliance-Control, Weather-Query, Calendar-Query, 
TVProgram-Play, Audio-Play, Other
"""


# ================= 2. 机器学习模型封装 =================

class IntentClassifierML:
    def __init__(self, data_path: str):
        """
        初始化机器学习分类器（KNN）
        :param data_path: 训练集 CSV 路径
        """
        self.vectorizer = CountVectorizer()
        self.model = KNeighborsClassifier()
        self._train(data_path)

    def _train(self, data_path: str):
        """加载数据并训练模型"""
        # 加载数据（假设第0列为文本，第1列为标签）
        dataset = pd.read_csv(data_path, sep="\t", header=None, nrows=10000)

        # 预处理：中文分词
        processed_text = dataset[0].apply(lambda x: " ".join(jieba.lcut(str(x))))

        # 特征提取
        x_features = self.vectorizer.fit_transform(processed_text)
        y_labels = dataset[1].values

        # 模型拟合
        self.model.fit(x_features, y_labels)
        print("ML Model training completed.")

    def predict(self, text: str) -> str:
        """
        使用机器学习进行分类
        """
        seg_text = " ".join(jieba.lcut(text))
        features = self.vectorizer.transform([seg_text])
        prediction = self.model.predict(features)
        return prediction[0]


# ================= 3. 大模型分类封装 =================

def text_classify_using_llm(text: str) -> str:
    """
    使用 LLM (Qwen) 进行零样本分类
    """
    prompt = f"""你是一个文本分类助手。请对以下用户输入进行分类。
输入文本：{text}

备选类别：[{CLASSIFICATION_LABELS}]

要求：仅输出选中的类别名称，不要返回任何多余的解释。"""

    try:
        completion = client.chat.completions.create(
            model="qwen-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Error: {str(e)}"


# ================= 4. 主程序入口 =================

if __name__ == "__main__":
    # 初始化 ML 模型
    # 确保目录下有 dataset.csv 文件
    if os.path.exists("dataset.csv"):
        ml_classifier = IntentClassifierML("dataset.csv")
    else:
        print("错误：未找到训练数据集 'dataset.csv'")
        ml_classifier = None

    # 测试用例
    test_query = "帮我导航到天安门"

    print(f"\n测试文本: '{test_query}'")
    print("-" * 30)

    # 1. 机器学习结果
    if ml_classifier:
        ml_result = ml_classifier.predict(test_query)
        print(f"机器学习分类结果: {ml_result}")

    # 2. 大模型结果
    llm_result = text_classify_using_llm(test_query)
    print(f"大语言模型分类结果: {llm_result}")
