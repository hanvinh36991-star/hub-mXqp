"""
作业1，本地配置es环境，也可以直接使用autodl镜像（账号id发给老师）， 学习基础的es操作。
"""
import time

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer


print("正在加载 SentenceTransformer 模型...")
model = SentenceTransformer('E:\AI学习\课程资料/models/BAAI/bge-small-zh-v1.5')
print("模型加载完成。")

# 1. 连接到Elasticsearch
def connect_elasticsearch():
    # 连接到本地Elasticsearch实例(默认端口 9200)
    es = Elasticsearch(
        "http://localhost:9200" # 默认址址和端口
    )

    # 测试连接
    if es.ping():
        print("连接 Elasticsearch 成功！")
        return es
    else:
        print("连接 Elasticsearch 失败！")
        return None

# 2. 创建索引
def create_index(es, index_name):
    """创建索引"""
    if not es.indices.exists(index = index_name):
        es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "category": {"type": "keyword"},  # 一级类目
                        "sub_category": {"type": "keyword"},  # 二级类目
                        "question": {"type": "text", "analyzer": "ik_max_word"},  # 问题（支持全文搜索，使用IK分词器）
                        "answer": {"type": "text", "analyzer": "ik_max_word"},  # 回答
                        "effective_time": {"type": "date"},  # 生效时间
                        "create_time": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"},  # 创建时间
                        "question_vector": {
                            "type": "dense_vector",
                            "dims": 512,
                            "index": True,
                            "similarity": "cosine"
                        } # 启用向量索引
                    }
                }
            }
        )
        print(f"索引{index_name}创建成功！")
    else:
        print(f"索引{index_name}已存在，跳过创建步骤。")

# 3. 添加单条文档
def add_docment(es, index_name, doc):
    """添加单条文档"""
    response = es.index(index=index_name, document=doc)
    print(f"文档添加成功, ID：{response['_id']}")
    return response['_id']

# 4. 批量添加文档
def bulk_add_documents(es, index_name, docs):
    """批量添加文档"""
    actions = []
    for doc in docs:
        action = {
            "_index": index_name,
            "_source": doc
        }
        actions.append(action)

    success, failed = bulk(es, actions)
    print(f"批量添加完成：成功{success}条，失败{len(failed)}条")
    return success

# 5. 全文搜索
def search_by_keyword(es, index_name, keyword, size=10):
    """根据关键词搜索文档"""
    query = {
        "query": {
            "match": {
                "question": keyword # 在问题字段中搜索
            }
        },
        "size": size
    }

    response = es.search(index=index_name, body=query)
    print(f"搜索 '{keyword}', 匹配到 {response['hits']['total']['value']} 条结果：")

    results = []
    for hit in response['hits']['hits']:
        result = {
            "id": hit["_id"],
            "score": hit['_score'],
            "question": hit['_source']['question'],
            "answer": hit["_source"]['answer'],
            "category": hit["_source"]["category"]
        }
        results.append(result)
        print(f"  得分: {hit['_score']:.2f}, 问题：{hit['_source']['question']}")

    return results

# 6. 按类目搜索
def search_by_category(es, index_name, category, sub_category=None, size=10):
    """根据类目搜索文档"""
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"category": category}}
                ]
            }
        },
        "size": size
    }

    # 如果指定了二级类目
    if sub_category:
        query['query']['bool']['must'].append({"term": {"sub_category": sub_category}})

    response = es.search(index=index_name, body=query)
    print(f"搜索类目 '{category}', 匹配到 {response['hits']['total']['value']} 条结果：")

    results = []
    for hit in response['hits']['hits']:
        result = {
            "id": hit["_id"],
            "question": hit['_source']['question'],
            "answer": hit["_source"]["answer"],
            "sub_category": hit["_source"].get("sub_category", '无')
        }
        results.append(result)
        print(f"   二级类目：{hit['_source'].get('sub_category', '无')}, 问题：{hit['_source']['question']}")

    return results

def search_by_knn(es, index_name, query_text, k=5, num_candidates=10):
    """根据文本向量相似度搜索文档(使用 KNN)"""
    query_vector = model.encode(query_text).tolist()

    # 构建 KNN 查询
    query = {
        "knn": {
            "field": "question_vector",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": num_candidates
        }
    }

    start_time = time.time()
    response = es.search(index=index_name, body=query)
    search_time = time.time() - start_time

    print(f"KNN搜索 '{query_text}', 匹配到 {len(response['hits']['hits'])} 条结果（耗时：{search_time:.4f}秒）：")

    results = []
    for hit in response['hits']['hits']:
        result = {
            "id": hit['_id'],
            "score": hit['_score'],
            "question": hit['_source']['question'],
            "answer": hit['_source']['answer'],
            "category": hit['_source']['category']
        }
        results.append(result)
        print(f"   得分: {hit['_score']:.4f}, 问题: {hit['_source']['question']}")

    return results

# 7. 更新文档
def update_document(es, index_name, doc_id, update_data):
    """更新文档"""
    update_body = {
        "doc": update_data
    }

    response = es.update(index=index_name, id=doc_id, body=update_body)
    print(f"文档 {doc_id} 更新成功！")
    return response

# 8. 删除文档
def delete_document(es, index_name, doc_id):
    """删除文档"""
    response = es.delete(index=index_name, id=doc_id)
    print(f"文档 {doc_id} 删除成功！")
    return response

# 9. 删除索引
def delete_index(es, index_name):
    """删除索引"""
    if es.indices.exists(index = index_name):
        es.indices.delete(index=index_name)
        print(f"索引 {index_name} 删除成功！")
    else:
        print(f"索引 {index_name} 不存在，跳过删除步骤。")

# 10. 获取索引信息
def get_index_info(es, index_name):
    """获取索引信息"""
    if es.indices.exists(index=index_name):
        info = es.indices.get(index=index_name)
        print(f"索引 {index_name} 信息：")
        print(f"    mappings：{info[index_name]['mappings']}")
        print(f"    settings：{info[index_name]['settings']}")
        return info
    else:
        print(f"索引 {index_name} 不存在。")
        return None

# 示例用法
if __name__ == "__main__":
    # 连接Elasticsearch
    es = connect_elasticsearch()

    if es:
        index_name = "faq"

        # 创建索引
        create_index(es, index_name)

        # 添加单条文档
        sample_doc = {
            "category": "账户问题",
            "sub_category": "登录问题",
            "question": "如何重置密码？",
            "answer": "请点击登录页面的 [忘记密码], 按照提示操作即可。",
            "effective_time": "2026-03-05T00:00:00",
            "create_time": "2026-03-05 10:00:00",
            "question_vector": model.encode("如何重置密码？").tolist()
        }
        doc_id = add_docment(es, index_name, sample_doc)

        # 批量添加文档
        bulk_docs = [
            {
                "category": "账户问题",
                "sub_category": "注册问题",
                "question": "如何注册新账户？",
                "answer": "访问官网，点击「注册」按钮，填写信息即可。",
                "effective_time": "2024-01-01T00:00:00",
                "create_time": "2024-01-01 10:01:00",
                "question_vector": model.encode("如何注册新账户？").tolist()
            },
            {
                "category": "产品问题",
                "sub_category": "功能使用",
                "question": "如何使用搜索功能？",
                "answer": "在页面顶部的搜索框中输入关键词，点击搜索按钮即可。",
                "effective_time": "2024-01-01T00:00:00",
                "create_time": "2024-01-01 10:02:00",
                "question_vector": model.encode("如何使用搜索功能？").tolist()
            },
            {
                "category": "订单问题",
                "sub_category": "退款问题",
                "question": "如何申请退款？",
                "answer": "在订单详情页点击「申请退款」，按照流程操作即可。",
                "effective_time": "2026-01-01T00:00:00",
                "create_time": "2026-01-01 10:03:00",
                "question_vector": model.encode("如何申请退款？").tolist()
            }
        ]
        bulk_add_documents(es, index_name, bulk_docs)

        # 搜索测试
        print("\n" + "="*50)
        search_by_keyword(es, index_name, "密码")

        print("\n" + "=" * 50)
        search_by_category(es, index_name, "账户问题")

        # 测试向量搜索
        print("\n" + "=" * 50)
        print("测试向量搜索(KNN)")
        test_queries = [
            "我忘记密码了怎么办？",
            "怎么注册账号？",
            "如何查找商品？"
        ]
        for query in test_queries:
            print(f"\n  查询: '{query}'")
            search_by_knn(es, index_name, query)


        # 更新文档
        print("\n" + "=" * 50)
        update_document(es, index_name, doc_id, {
            "answer": "请点击登录页面的「忘记密码」，按照提示操作即可。如有问题，请联系客服。",
            "effective_time": "2025-02-01T00:00:00"
        })

        # 删除文档
        print("\n" + "=" * 50)
        delete_document(es, index_name, doc_id)

        # 获取索引信息
        print("\n" + "=" * 50)
        get_index_info(es, index_name)

        # 删除索引（确认操作）
        print("\n" + "=" * 50)
        # delete_index(es, index_name)

        print("\n✅ 所有操作完成！")
