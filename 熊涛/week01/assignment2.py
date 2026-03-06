from idlelib.pyparse import trans
from sys import prefix

import jieba
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

df = pandas.read_csv("dataset.csv", names=['text','type'],sep='\t', nrows=None, encoding='utf-8')
# print(dataset['text'].head(5))
jieba.add_word("和平精英")
jieba.add_word("墓王之王")
jieba1 = lambda x: " ".join(jieba.lcut(x,  cut_all=True))
train_data = df['text'].apply(jieba1)
# print(df['type'].value_counts())
# print(train_data.head(10))
vector = CountVectorizer()

input_feature = vector.fit_transform(train_data.values)

model = KNeighborsClassifier()
model.fit(input_feature, df['type'].values)
# print(model)

def to_ml(text: str) -> str:
    prompt = vector.transform([jieba1(text)])

    return model.predict(prompt)

def to_llm(text: str) -> str:
    client = OpenAI(
        api_key='sk-1db02ffe72144cc6bfe3500437314ffd',
        base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    )

    messages = [{'role': 'system','content': f'通过分析用户的文本，返回文本种类中的一个{df['type'].unique()}]'}]
    messages.append({'role': 'user', 'content': text})

    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=messages,
        stream=False
    )

    result = completion.choices[0].message.content.strip()
    return result


if __name__ == '__main__':
    print(to_ml("帮我打开和平精英下飞机的音乐"))
    print(to_llm("帮我打开和平精英下飞机的音乐"))