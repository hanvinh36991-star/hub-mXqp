import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

dataset=pd.read_csv('dataset.csv',sep='\t',header=None,nrows=10000)
input_sentence=dataset[0].apply(lambda row: ' '.join(jieba.lcut(str(row))))
vector=CountVectorizer()
vector.fit(input_sentence.values)
input_feature=vector.transform(input_sentence.values)

model=KNeighborsClassifier()
model.fit(input_feature,dataset.iloc[:,1])

def text_classify_using_ml(text: str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]


from openai import OpenAI

client = OpenAI(
        api_key="sk-fab60036xxxxx15827702c5813",

        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def text_classify_using_llm(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类:{text}
输出的类别只能从如下中进行选择,只输出类别
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


if __name__ == '__main__':
    print('机器学习：',text_classify_using_ml('帮我导航到天安门'))
    print('大语言模型：',text_classify_using_llm('帮我播放电影流浪地球'))
