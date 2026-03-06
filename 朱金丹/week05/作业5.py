import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
 
# ====================== 1. 生成测试数据======================
np.random.seed(42)
data1 = np.random.normal(loc=[0, 0], scale=0.3, size=(50, 2))  # 紧凑
data2 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))  # 一般
data3 = np.random.normal(loc=[-1, 3], scale=0.8, size=(50, 2)) # 松散
X = np.vstack([data1, data2, data3])
 
# ====================== 2. KMeans 聚类 ======================
k = 3  # 聚类数
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)  # 每个样本的类别
centers = kmeans.cluster_centers_  # 聚类中心
 
# ====================== 3. 计算每个类的【类内平均距离】=====================
def cluster_inner_distance(X, labels, cluster_id):
    """计算某个类 cluster_id 的样本到中心的平均距离"""
    cluster_points = X[labels == cluster_id]
    center = centers[cluster_id]
    distances = np.linalg.norm(cluster_points - center, axis=1)
    return np.mean(distances)
 
# 对每个类别计算类内距离
inner_distances = {}
for c in range(k):
    inner_distances[c] = cluster_inner_distance(X, labels, c)
 
# ====================== 4. 按类内距离【从小到大排序】=====================
# 越小越紧凑，排在前面
sorted_clusters = sorted(inner_distances.items(), key=lambda x: x[1])
 
# ====================== 5. 输出结果 ======================
print("每个类的类内平均距离：")
for c, d in inner_distances.items():
    print(f"类别 {c}：{d:.4f}")
 
print("\n按类内距离从小到大排序（越靠前越紧凑）：")
for i, (c, d) in enumerate(sorted_clusters):
    print(f"第 {i+1} 名：类别 {c}，类内距离 {d:.4f}")
