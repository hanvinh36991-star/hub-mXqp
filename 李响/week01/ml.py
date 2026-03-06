import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

dataset=pd.read_csv("dataset.csv",sep="\t",header=None,nrows=10000)
print( dataset.head(9))
print(dataset[1].value_counts())

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

vector=CountVectorizer()
vector=vector.fit(input_sententce.values)
input_vector=vector.transform(input_sententce.values)

model=KNeighborsClassifier()
model.fit(input_vector,dataset[1].values)

def text_classify_using_knn(text: str) -> str:
    test_sentence=" ".join(jieba.lcut(text))
    test_vector=vector.transform([test_sentence])
    return model.predict(test_vector)[0]
print(text_classify_using_knn("帮我找一个游戏"))


print("机器学习:",text_classify_using_knn("帮我导航到小吃街"))
