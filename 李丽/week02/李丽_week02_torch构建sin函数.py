
import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt #用于绘图

# 1. x修改为 -5 到 5 之间的 100 个点（reshape(-1, 1) 转换为二维矩阵），y 为对应的正弦值，加入随机噪声
X_numpy = np.linspace(-5, 5, 100).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) * 0.1 # 修改为正弦函数，加入随机噪声
X = torch.from_numpy(X_numpy).float() # 将numpy数组转换为 torch 张量，float（）是为了精度对齐。numpy默认使用双精度，torch默认使用单精度
y = torch.from_numpy(y_numpy).float()


# 2. 定义模型
class SimpleClassifier(torch.nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.net=torch.nn.Sequential(
            torch.nn.Linear(1,50),
            torch.nn.Tanh(),
            torch.nn.Linear(50,1)# 接收上一层的 50 个特征，输出 1 个结果
        )

    def forward(self, x):
        return self.net(x)


# 3. 训练与可视化
loss_fn = torch.nn.MSELoss()
model=SimpleClassifier() # 实例化
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 优化器，基于模型参数 梯度 自动更新

# 4. 训练模型
num_epochs = 1000 #可以理解为训练多少轮
for epoch in range(num_epochs):

    y_pred=model(X)

    # 计算损失
    #模型输出VS真实标签
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加，默认不会清空历史
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数 在这一步会更新

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！") # 查看训练的效果

print("---" * 10)

# 6. 绘制结果
with torch.no_grad():
    y_predicted = model(X).detach().numpy() 

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label='Neural Network Fitting', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
