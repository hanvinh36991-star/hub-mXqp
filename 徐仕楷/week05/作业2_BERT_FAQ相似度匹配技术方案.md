# 作业2：BERT文本编码与相似度计算技术方案

## 一、技术方案流程图

### 1. 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           离线处理阶段                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐           │
│   │ FAQ数据  │ ──► │ 数据清洗 │ ──► │ BERT编码 │ ──► │ 向量存储 │           │
│   │          │     │          │     │          │     │          │           │
│   │ • 标题   │     │ • 分词   │     │ • 预训练 │     │ • Milvus │           │
│   │ • 相似问 │     │ • 过滤   │     │ • 768维  │     │ • FAISS  │           │
│   │ • 答案   │     │ • 规范化 │     │ • 池化   │     │          │           │
│   └──────────┘     └──────────┘     └──────────┘     └──────────┘           │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           在线服务阶段                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐           │
│   │ 用户提问 │ ──► │ BERT编码 │ ──► │ 向量检索 │ ──► │ 返回答案 │           │
│   │          │     │          │     │          │     │          │           │
│   │ 用户输入 │     │ 实时编码 │     │ Top-K    │     │ 置信度   │           │
│   │ 的问题  │     │          │     │ 余弦相似 │     │ 阈值过滤 │           │
│   └──────────┘     └──────────┘     └──────────┘     └──────────┘           │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2. FAQ离线编码流程图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        FAQ离线编码流程                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 步骤1：数据准备                                                      │     │
│  │      ├── 读取FAQ数据库                                               │     │
│  │      ├── 获取FAQ标题                                                 │     │
│  │      ├── 获取相似提问（每条FAQ最多200条）                             │     │
│  │      └── 获取答案内容                                                │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 步骤2：文本预处理                                                    │     │
│  │      ├── 分词：使用BERT Tokenizer                                     │     │
│  │      ├── 添加特殊标记：[CLS]（句子开头）、[SEP]（句子结尾）            │     │
│  │      ├── 长度处理：max_length=128（超长截断）                         │     │
│  │      └── 编码：转换为token_id、attention_mask                         │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 步骤3：BERT模型推理                                                  │     │
│  │      ├── 加载预训练模型：bert-base-chinese                            │     │
│  │      ├── 输入：tokenized数据                                          │     │
│  │      ├── 前向传播：计算last_hidden_state                              │     │
│  │      └── 池化：取[CLS]位置的向量作为语义表示                         │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 步骤4：向量存储                                                      │     │
│  │      ├── 归一化：对向量进行L2归一化（加速余弦相似计算）                │     │
│  │      ├── 存入向量数据库                                               │     │
│  │      │   ├── Milvus（生产环境推荐）                                   │     │
│  │      │   └── FAISS（单机测试）                                        │     │
│  │      └── 建立索引：IVF/HNSW等（支持亿级向量毫秒级检索）               │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3. 用户在线匹配流程图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        用户在线匹配流程                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 用户发起提问                                                          │     │
│  │ 例："我的订单还没发货，怎么回事？"                                     │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 步骤1：文本预处理                                                     │     │
│  │      ├── 使用BERT Tokenizer分词                                       │     │
│  │      ├── 添加[CLS]和[SEP]标记                                         │     │
│  │      └── 转换为token_id、attention_mask                                │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 步骤2：BERT在线编码                                                   │     │
│  │      ├── 输入：用户问题经过预处理的token序列                          │     │
│  │      ├── 模型推理：前向传播计算隐藏状态                               │     │
│  │      └── 输出：[CLS]位置的768维语义向量                               │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 步骤3：向量检索（ANN搜索）                                            │     │
│  │      ├── 搜索策略：余弦相似度                                        │     │
│  │      ├── Top-K检索：返回最相似的K条FAQ（通常K=3~5）                  │     │
│  │      └── 检索库：在Milvus/FAISS中搜索                                │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 步骤4：相似度计算                                                    │     │
│  │                                                                         │     │
│  │      使用余弦相似度公式：                                              │     │
│  │      ┌─────────────────────────────────────────────────────────┐       │     │
│  │      │                                                         │       │     │
│  │      │   cos(A, B) = (A · B) / (||A|| × ||B||)               │       │     │
│  │      │                                                         │       │     │
│  │      │   其中：                                                  │       │     │
│  │      │   A = 用户问题向量                                        │       │     │
│  │      │   B = FAQ向量                                            │       │     │
│  │      │                                                         │       │     │
│  │      └─────────────────────────────────────────────────────────┘       │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 步骤5：结果筛选与返回                                                │     │
│  │      ├── 相似度阈值：设定最低匹配分数（如0.7）                        │     │
│  │      ├── 过滤低分结果：低于阈值的FAQ不返回                           │     │
│  │      └── 按相似度从高到低返回                                        │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 返回答案                                                              │     │
│  │                                                                         │     │
│  │   ✅ 最佳匹配FAQ：订单发货问题                                         │     │
│  │      相似度：0.85                                                    │     │
│  │      答案：亲，订单正在处理中，预计2-3天发货，请耐心等待~             │     │
│  │                                                                         │     │
│  │   📝 如果所有FAQ相似度都低于阈值：                                     │     │
│  │      返回："未能找到相关答案，请联系人工客服"                         │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、BERT编码详细原理

### 1. BERT模型结构

```
输入层： [CLS] 你 好 吗 ？ [SEP]
         ↓
Token Embedding：每个词转换为词向量
         ↓
Position Embedding：位置编码（表示词在句子中的位置）
         ↓
Segment Embedding：区分上下文的句子对
         ↓
BERT Encoder（12层Transformer）
         ├── Multi-Head Attention（多头注意力）
         ├── Add & Norm（残差连接+层归一化）
         ├── Feed Forward（前馈神经网络）
         └── Add & Norm
         ↓
输出层： [CLS]向量 [词1]向量 [词2]向量 ... [SEP]向量
         ↓
池化：取[CLS]位置的向量作为整个句子的语义表示
```

### 2. 编码步骤详解

**步骤1：Tokenization（分词）**

使用BERT自带的WordPiece分词器：

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

text = "我的订单还没发货"
tokens = tokenizer.tokenize(text)
# 输出：['我', '的', '订', '单', '还', '没', '发', '货']
```

**步骤2：添加特殊标记**

```
[CLS] 我的订单还没发货 [SEP]
  ↑                    ↑
句子开头              句子结尾
```

**步骤3：Padding和Truncation**

将不同长度的文本填充或截断到统一长度：

```
[CLS] 我 的 订 单 还 没 发 货 [SEP] [PAD] [PAD] ...
```

**步骤4：转换为ID**

```
tokens = ['[CLS]', '我', '的', '订', '单', '还', '没', '发', '货', '[SEP]', '[PAD]']
token_ids = [101, 2769, 1168, 712, 981, 6826, 697, 2207, 5445, 102, 0]
attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  # 1=真实token，0=padding
```

**步骤5：BERT推理**

```python
import torch
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-chinese')
inputs = {
    'input_ids': torch.tensor([token_ids]),
    'attention_mask': torch.tensor([attention_mask])
}

with torch.no_grad():
    outputs = model(**inputs)

# last_hidden_state shape: [batch_size, seq_len, hidden_size]
# [CLS]位置是索引0
cls_vector = outputs.last_hidden_state[:, 0, :]  # shape: [1, 768]
```

**步骤6：向量池化**

```python
# 方法1：直接使用[CLS]位置的向量
cls_vector = outputs.last_hidden_state[:, 0, :]  # 推荐方法

# 方法2：Mean Pooling（所有词向量取平均）
mean_vector = torch.mean(outputs.last_hidden_state, dim=1)

# 方法3：Max Pooling（所有词向量取最大）
max_vector = torch.max(outputs.last_hidden_state, dim=1).values
```

---

## 三、余弦相似度计算详解

### 1. 数学公式

```
余弦相似度 = 点积 / (向量A的范数 × 向量B的范数)

cos(A, B) = (A · B) / (||A|| × ||B||)
```

### 2. 公式分解

**点积（A · B）**

```
A · B = Σ(Ai × Bi) = A1×B1 + A2×B2 + ... + An×Bn

例如：
A = [0.1, 0.5, 0.8, 0.3]
B = [0.2, 0.4, 0.7, 0.5]

A · B = 0.1×0.2 + 0.5×0.4 + 0.8×0.7 + 0.3×0.5
      = 0.02 + 0.20 + 0.56 + 0.15
      = 0.93
```

**L2范数（||A||）**

```
||A|| = √(A1² + A2² + ... + An²)

例如：
A = [0.1, 0.5, 0.8, 0.3]

||A|| = √(0.1² + 0.5² + 0.8² + 0.3²)
     = √(0.01 + 0.25 + 0.64 + 0.09)
     = √0.99
     = 0.995
```

**完整计算**

```
B = [0.2, 0.4, 0.7, 0.5]

||B|| = √(0.2² + 0.4² + 0.7² + 0.5²)
     = √(0.04 + 0.16 + 0.49 + 0.25)
     = √0.94
     = 0.970

cos(A, B) = 0.93 / (0.995 × 0.970)
          = 0.93 / 0.965
          = 0.96

结论：相似度为0.96，表示高度相似
```

### 3. 相似度含义

| 值范围 | 含义 |
|-------|------|
| 0.9 ~ 1.0 | 非常相似 |
| 0.7 ~ 0.9 | 比较相似 |
| 0.5 ~ 0.7 | 部分相似 |
| 0.3 ~ 0.5 | 相似度较低 |
| 0.0 ~ 0.3 | 基本不相关 |
| < 0 | 语义相反 |

---

## 四、核心代码实现

### 1. BERT编码

```python
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

class BERTEncoder:
    def __init__(self, model_name='bert-base-chinese'):
        """
        初始化BERT编码器
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
    
    def encode(self, text, max_length=128):
        """
        对文本进行BERT编码
        参数：
            text: 输入文本
            max_length: 最大序列长度
        返回：
            vector: 768维语义向量
        """
        # 分词和编码
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
        
        # BERT推理
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 取[CLS]位置的向量
        cls_vector = outputs.last_hidden_state[:, 0, :]
        
        return cls_vector.squeeze().numpy()
    
    def encode_batch(self, texts, batch_size=32):
        """
        批量编码
        """
        all_vectors = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                max_length=128,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            batch_vectors = outputs.last_hidden_state[:, 0, :].numpy()
            all_vectors.append(batch_vectors)
        
        return np.vstack(all_vectors)
```

### 2. 相似度计算

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityCalculator:
    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """
        计算两个向量的余弦相似度
        """
        vec_a = vec_a.reshape(1, -1)
        vec_b = vec_b.reshape(1, -1)
        return cosine_similarity(vec_a, vec_b)[0][0]
    
    @staticmethod
    def cosine_similarity_matrix(vectors_a, vectors_b):
        """
        计算两个向量矩阵之间的余弦相似度
        """
        return cosine_similarity(vectors_a, vectors_b)
```

### 3. 完整匹配流程

```python
class FAQMatcher:
    def __init__(self, model_name='bert-base-chinese'):
        self.encoder = BERTEncoder(model_name)
        self.faqs = []
        self.vectors = []
    
    def build_index(self, faq_list):
        """
        构建FAQ向量索引（离线）
        """
        texts = [faq['title'] for faq in faq_list]
        vectors = self.encoder.encode_batch(texts)
        
        self.faqs = faq_list
        self.vectors = vectors
        
        print(f"✅ 已构建索引，包含 {len(faq_list)} 条FAQ")
    
    def search(self, query, top_k=3, threshold=0.7):
        """
        在线检索
        """
        # 编码查询
        query_vector = self.encoder.encode(query)
        
        # 计算相似度
        similarities = []
        for i, faq_vec in enumerate(self.vectors):
            sim = SimilarityCalculator.cosine_similarity(query_vector, faq_vec)
            similarities.append((i, sim))
        
        # 排序和筛选
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, sim in similarities[:top_k]:
            if sim >= threshold:
                results.append({
                    'faq': self.faqs[idx],
                    'similarity': sim
                })
        
        return results
```

---

## 五、总结

| 阶段 | 输入 | 处理 | 输出 |
|-----|------|------|------|
| 离线 | FAQ数据 | BERT编码 | 768维向量 |
| 存储 | 向量 | 存入向量库 | Milvus/FAISS |
| 在线 | 用户提问 | BERT编码 | 查询向量 |
| 检索 | 查询向量 | 余弦相似度 | Top-K FAQ |

**关键技术点：**
1. 使用bert-base-chinese预训练模型
2. 取[CLS]位置的向量作为语义表示
3. 使用余弦相似度计算文本相似度
4. 使用向量数据库支持高效检索
