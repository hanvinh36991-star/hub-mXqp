import pandas as pd
import jieba

from openai import OpenAI

dataset = pd.read_csv('dataset.csv', sep="\t", names=['text', 'Category'], header=None)

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-e1b4d6b63d0xxxxxb34e1fd2d72e",

    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def query(text: str) -> str:
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
            """},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    #取任意一条数据进行访问
    result = query(dataset['text'][0])
    print(result)

