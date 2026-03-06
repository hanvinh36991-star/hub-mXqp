import pandas as pd
import jieba

from sklearn.feature_extraction.text import CountVectorizer
from openai import OpenAI
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)

# sklearn对中文处理
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

vector = CountVectorizer()
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

client = OpenAI(
    api_key = "sk-7458206891xxxx46d6f7366fecdd5",
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def text_classify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    :param text: 输入提示词
    :return: 推理结果
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]


def text_classify_using_llm(text: str) -> str:
    """
    文本分类(大语言模型)，输入文本完成类型划分
    """
    completion = client.chat.completions.create(
        # 模型名称
        model="qwen-plus",
        messages=[
            {"role": "user", "content": f""" 帮我进行文本分类：{text}
            
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
"""},
        ]
    )
    return completion.choices[0].message.content

if __name__=="__main__":
    print("机器学习：", text_classify_using_ml("帮我导航到天安门"))
    print("大语言模型：", text_classify_using_llm("帮我导航到天安门"))
