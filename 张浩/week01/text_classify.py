import os
from openai import OpenAI
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# 特征提取
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
# print(dataset[1].value_counts())
labels = dataset[1].unique()
# print(labels)


def text_calssify_using_ml(text: str) -> str:
    input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # sklearn对中文处理

    vector = CountVectorizer()  # 对文本进行提取特征 默认是使用标点符号分词
    vector.fit(input_sententce.values)  # 统计词表
    input_feature = vector.transform(input_sententce.values)  # 进行转换 100 * 词表大小

    # KNN模型训练
    model = KNeighborsClassifier()
    model.fit(input_feature, dataset[1].values)

    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def text_calssify_using_llm(text: str) -> str:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "user", "content": text},
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    text = '今天是周几'
    print("机器学习: ", text_calssify_using_ml(text))
    print("大语言模型: ", text_calssify_using_llm(f"""帮我进行文本分类：{text}。
    输出的类别只能从如下中进行选择:{labels},不要添加其他字词"""))
