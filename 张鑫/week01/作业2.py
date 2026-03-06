import jieba # 中文分词库
import pandas as pd # 数据转换库
from sklearn.feature_extraction.text import CountVectorizer # 特征提取器 : 将文本转换成词频矩阵
from sklearn.neighbors import KNeighborsClassifier # 分类算法模型 : KNN近邻算法


# 1. 加载数据
datasets = pd.read_csv("dataset.csv", sep="\t", header=None) # 读取 CSV 文件并转换为 DataFrame


# 2. 数据预处理
input_sentence = datasets[0].apply(lambda x: " ".join(jieba.lcut(x))) # 将datasets的第一列通过空格将中文分词放到input_sentence


# 3. 特征工程 (将文本数字化)
vector = CountVectorizer() # 初始化词频统计工具
vector.fit(input_sentence.values) # 第一步、构建一个词汇表
input_feature = vector.transform(input_sentence.values) # 第二步、根据词汇表转换成数字矩阵


# 4. 模型训练 (Machine Learning)
model = KNeighborsClassifier() # 初始化 KNN 分类器实例
model.fit(input_feature, datasets[1].values) # 让模型学习特征（输入）与标签（答案）之间的对应关系


def analyze_sentiment_ml(text: str) -> str:
    """
    使用 KNN 模型对输入文本进行分析。

    Args:
        text: 用户输入的原始文本。

    Returns:
        str: 预测出的类别标签。
    """
    test_sentence = " ".join(jieba.lcut(text)) # 将用户输入的语句进行分词
    test_feature = vector.transform([test_sentence]) # 将新句子转换为与训练集同纬度的数字向量
    return model.predict(test_feature)[0] # 获取预测结果列表中的第一个元素


from openai import OpenAI # 调用openai库重命名给OpenAI


client = OpenAI( # 初始化客户端
    api_key="---------------------", # 输入api_key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 输入厂商的地址
)


def analyze_sentiment_llm(text: str) -> str:
    """
    调用 Qwen 大模型进行文本分类。

    注意：此函数会产生 API 费用，且受网络延迟影响

    Args:
        text: 待分类的自然语言文本

    Returns:
        str: 模型返回的清洗后的分类标签（已去除多余的解释性文字）
    """
    completion = client.chat.completions.create( # 创建一个客户端的对话
        model="qwen-flash",  # 模型
        messages=[ # 构建 Prompt ，包括上下文和指令
            {
                "role": "user", "content": f"""帮我进行文本分类：{text}
输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。你只只负责输出分类标签，禁止输出任何其他废话、标点符号或解释。
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
"""
            }
        ]
    )
    return completion.choices[0].message.content # 返回预测的结果



if __name__ == "__main__":
    print("机器学习结果: ", analyze_sentiment_ml("我要去北京天安门")) # 用机器学习分析这句话
    print("机器学习结果: ", analyze_sentiment_ml("物流很快，质量超好，好评！")) # 用机器学习分析这句话
    print("机器学习结果: ", analyze_sentiment_ml("到底去水果店北京地方吃个酸辣粉看个演唱会听杰伦唱歌")) # 用机器学习分析这句话
    print("机器学习结果: ", analyze_sentiment_ml("我今天起飞到了北京，吃了个酸辣粉，又去看了一场演唱会，又去吃了牛腩粉，后面回到了深圳")) # 用机器学习分析这句话
    print("LLM 结果:", analyze_sentiment_llm("我要去北京天安门")) # 用大语言模型分析这句话
    print("LLM 结果:", analyze_sentiment_llm("物流很快，质量超好，好评！")) # 用大语言模型分析这句话
    print("LLM 结果:", analyze_sentiment_llm("到底去水果店北京地方吃个酸辣粉看个演唱会听杰伦唱歌")) # 用大语言模型分析这句话
    print("LLM 结果:", analyze_sentiment_llm("我今天起飞到了北京，吃了个酸辣粉，又去看了一场演唱会，又去吃了牛腩粉，后面回到了深圳")) # 用大语言模型分析这句话