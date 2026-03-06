import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import jieba
from openai import OpenAI


# 读取数据集文件
# sep -- 列分割符， header -- 指定表头，默认将第一行当做表头【行索引为0】，由于当前的数据集没有表头，设置为None , nrows -- 读取的行数
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)


# dataset[1] - 访问列名为 1的列 , 由于header是None, 列名默认为0，1，2...
# dataset.iloc[:, 1] - 访问第2列（位置索引）
# print(dataset.iloc[:, 1].value_counts())

#dataset 进行jieba分词, 分词后结果用空格连接成一个字符串
imputData = dataset.iloc[:, 0].apply(lambda x: " ".join(jieba.cut(x)))

# sklearn 搭配 jieba的中文分词，对文本进行向量化统计
# 导入 sklearn 的 CountVectorizer 类 -- 统计词频
vectorizer = CountVectorizer()

# 对文本进行向量化统计
# vectorizer.fit_transform()是 sklearn 中文本向量化的核心方法，它将文本转换为机器学习模型可用的数值特征矩阵。
# fit -- 统计词频
# transform -- 文本进行向量化统计
# fit_transform -- 统计词频并对文本进行向量化统计
X = vectorizer.fit_transform(imputData.values)

# Knn 分类器
# 导入 sklearn 的 KNeighborsClassifier 类 -- 分类器

# KNeighborsClassifier是 sklearn 中的 K最近邻（K-Nearest Neighbors）分类器，n_neighbors=3表示使用 3个最近邻居​ 进行投票决策。
knn = KNeighborsClassifier(n_neighbors=3)
# 训练分类器
knn.fit(X, dataset.iloc[:, 1])

# 对新文本进行分类预测

def predict_tokenized_ml(text:str) -> str :
    # 对新文本进行分词
    tokenized_text = " ".join(jieba.cut(text))
    # 对分词后的文本进行向量化
    vectorized_text = vectorizer.transform([tokenized_text])
    # 使用训练好的分类器进行预测
    predicted_class = knn.predict(vectorized_text)
    return predicted_class[0]

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-8366xxxxxc58be703604",

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def predict_tokenized_llm(imputWord: str)-> str:
    response = client.chat.completions.create(
        model="qwen-flash",
        # f"""是 Python 中三重引号格式化字符串的语法，结合了多行字符串和格式化字符串的功能。{imputWord}表示变量值
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{imputWord}

输出的类别只能从如下中进行选择， 除了类别之外不要输出其他内容，请给出最合适的类别。
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
    return response.choices[0].message.content


# main
if __name__ == "__main__":
    predicted_class = predict_tokenized_ml("我想听一首陈奕迅的《陀飞轮》")
    print("ml预测类别:", predicted_class)

    predicted_class = predict_tokenized_llm("我想听一首陈奕迅的《陀飞轮》")
    print("llm预测类别:", predicted_class)
