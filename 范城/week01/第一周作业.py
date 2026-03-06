import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

file_path = '/Users/fancheng/八斗学习/Week01/dataset.csv'
dataset = pd.read_csv(file_path, sep='\t', header=None, nrows=None)

def text_classify_using_sklearn(text: str) -> str:
    input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

    vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
    vector.fit(input_sententce.values)
    input_feature = vector.transform(input_sententce.values)

    model = KNeighborsClassifier()
    model.fit(input_feature, dataset[1].values)

    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)

def text_classify_using_qwen(text: str) -> str:
    client = OpenAI(api_key="sk-81d03a1beaxxxxx6c5512b1b6", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    response = client.chat.completions.create(
        model="qwen3-max",
        messages=[{"role": "user", "content": f""":请进行文本分类{text}
        输出的类别只能从如下类别中选择：
        FilmTele-Play
        Video-Play
        Music-Play
        Radio-Listen
        Alarm-Update
        Weather-Query
        Travel-Query
        HomeAppliance-Control
        Calendar-Query
        TVProgram-Play
        Audio-Play
        Other
        """}]
    )
    return response.choices[0].message.content

def text_classify_using_deepseek(text: str) -> str:
    client = OpenAI(api_key="sk-ed8596837xxxx1406a13c8841", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": f""":请进行文本分类{text}
        输出的类别只能从如下类别中选择：
        FilmTele-Play
        Video-Play
        Music-Play
        Radio-Listen
        Alarm-Update
        Weather-Query
        Travel-Query
        HomeAppliance-Control
        Calendar-Query
        TVProgram-Play
        Audio-Play
        Other
        """}]
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    text = "今天天气如何"
    print("KNN模型预测结果: ",text_classify_using_sklearn(text))
    print("千问大模型预测结果: ",text_classify_using_qwen(text))
    print("DeepSeek大模型预测结果: ",text_classify_using_deepseek(text))
