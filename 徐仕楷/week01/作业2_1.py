import jieba # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN

# pip install openai
from openai import OpenAI


# 数据读取
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=12100)
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

# print(dataset[1].value_counts())
# 特征提取
vector = CountVectorizer()
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)
# 模型加载&数据训练
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

def text_calssify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    print(f"对话输入: {text}")
    test_query = text
    test_sentence = " ".join(jieba.lcut(test_query))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def text_calssify_using_llm(text: str) -> str:
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        # https://bailian.console.aliyun.com/?tab=model#/api-key
        api_key="sk-7e56xxxxxx3d794b8e0e9fe522162",  # 账号绑定，用来计费的

        # 大模型厂商的地址，阿里云
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-flash",  # 模型的代号

        # 对话列表
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},  # 给大模型的命令，角色的定义
            {"role": "user", "content": "你好？"},  # 用户的提问
            {"role": "user", "content": "你好？"},  # 用户的提问
            {"role": "user", "content": "你回家吃饭吗，宝贝？"},  # 用户的提问
        ]
    )

    return completion.choices[0].message.content


def using_ml_test():
    ret = text_calssify_using_ml("帮我播放周杰伦的音乐")
    print(f"对话输出: {ret}")
    ret = text_calssify_using_ml("我要吃饭")
    print(f"对话输出: {ret}")
    ret = text_calssify_using_ml("我需要休息")
    print(f"对话输出: {ret}")
    ret = text_calssify_using_ml("帮我打扫卫生")
    print(f"对话输出: {ret}")
    ret = text_calssify_using_ml("帮我播放周星驰的电影")
    print(f"对话输出: {ret}")

def using_using_llm():
    ret = text_calssify_using_ml("帮我播放周杰伦的音乐")
    print(f"对话输出: {ret}")
    ret = text_calssify_using_ml("我要吃饭")
    print(f"对话输出: {ret}")
    ret = text_calssify_using_ml("我需要休息")
    print(f"对话输出: {ret}")
    ret = text_calssify_using_ml("帮我打扫卫生")
    print(f"对话输出: {ret}")
    ret = text_calssify_using_ml("帮我播放周星驰的电影")
    print(f"对话输出: {ret}")

def main():
    using_ml_test()
    print()
    using_using_llm()

if __name__ == "__main__":
    main()
