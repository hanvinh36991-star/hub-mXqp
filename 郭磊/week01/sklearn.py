import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 加载数据
dataset = pd.read_csv('dataset.csv', sep="\t", names=['text', 'Category'], header=None)

# sklearn对中文处理
X = dataset['text'].apply(lambda x: " ".join(jieba.lcut(x)))
y = dataset.copy().drop(['text'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)

vector = CountVectorizer()

# 统计词表
vector.fit(X_train.values)

# 100 * 词表大小
inputFeature = vector.transform(X_train.values)

model = KNeighborsClassifier()

model.fit(inputFeature, y_train.values)

if __name__ == '__main__':
    for idx in X_test.index:
        testQuery = X_test[idx]
        print("输入文本为:", testQuery)
        testSentence = " ".join(jieba.lcut(testQuery))
        tf = vector.transform([testSentence])
        preResult = model.predict(tf)
        print("预期结果为:", y_test['Category'][idx])
        print("实际结果为:", preResult[0])
        print("========================")
