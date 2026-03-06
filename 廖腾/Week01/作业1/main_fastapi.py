## 11_文本分类
import jieba
from openai import OpenAI
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from fastapi import FastAPI

app = FastAPI()
dataset=pd.read_csv("./dataset.csv",sep="\t",header=None,nrows=None)
input_sentence=dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))##中文处理
vector = CountVectorizer()##对文本进行提取特征，默认使用标点符号分词，不是模型
vector.fit(input_sentence.values)##统计词表
input_feature = vector.transform(input_sentence.values)#100*词表大小
model = KNeighborsClassifier()#Knn模型训练
model.fit(input_feature,dataset[1].values)

@app.get("/text-cls/ml")
def text_classify_ml(text:str)-> str:
    """
    文本分类(机器学习)，输入文本实现类别划分
    """
    test_sentence=" ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

@app.get("/text-cls/llm")
def text_classify_llm(text:str)-> str:
    """
    文本分类(大模型)，输入文本实现类别划分
    """
    client = OpenAI(
    api_key="sk-bdfb5891d156408bb2f6a1846aa7f6c5",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
,
    )
    completion=client.chat.completions.create(
    model="qwen-flash",
    messages=[
        {"role": "user","content":
            f"""帮我进行文本分类：{text}
            输出的类别只能从如下进行选择：
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
                Other  """},]
    )
    return completion.choices[0].message.content

if __name__=='__main__':
    print(text_classify_ml("帮我播放一下郭德纲的小品"))
    print(text_classify_llm("帮我导航到天安门"))