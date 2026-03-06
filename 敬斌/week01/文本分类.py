import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

data = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
# 分词处理
input_sentence = data[0].apply(lambda x: " ".join(jieba.lcut(x)))

vectorizer = CountVectorizer()
vectorizer.fit(input_sentence.values)
transform = vectorizer.transform(input_sentence.values)

knn = KNeighborsClassifier()
knn.fit(transform, data[1].values)

client = OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-46620xxxxx90804bcfecaa",

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def test_classify_using_ml(text: str) -> str:
    """
    使用机器学习进行文本分类
    :param text:
    :return:
    """
    test_sentence = " ".join(jieba.lcut(text))
    transform = vectorizer.transform([test_sentence])
    return knn.predict(transform)[0]

def test_classify_using_llm(text: str) -> str:
    """
    使用大语言模型进行文本分类
    :param text:
    :return:
    """
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}
            输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
            FilmTele-Play            
            Video-Play               
            Music-Play              
            Radio-Listen           
            Alarm-Update        
            Travel-Query        
            HomeAppliance-Control  
            Weather-Query          
            Calendar-Query      
            TVProgram-Play      
            Audio-Play       
            Other             
            """}
        ]
    )
    return completion.choices[0].message.content

if __name__ == '__main__':
    print("机器学习分类结果：" + test_classify_using_ml("我想去吃兰州拉面"))
    print("大语言模型分类结果：" + test_classify_using_ml("我想去吃兰州拉面"))

