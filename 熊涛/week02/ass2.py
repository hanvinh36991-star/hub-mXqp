import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 10
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(100, 1)


X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)



class SimpleClassifier(torch.nn.Module):
    def __init__(self, hidden_dim_1, hidden_dim_2):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim_1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_1,hidden_dim_2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_dim_2, 1)
        )

    def forward(self, x):
        return self.network(x)

model = SimpleClassifier(64,64)


loss_fn = torch.nn.MSELoss() # 回归任务


optimizer = torch.optim.SGD(model.parameters(), lr=0.05) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    model.train()
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    X_test_numpy = np.linspace(0, 10, 500).reshape(-1, 1)

    X_test_tensor = torch.from_numpy(X_test_numpy).float()
    y_test_pred = model(X_test_tensor).numpy()

plt.figure(figsize=(10, 6))
# 画出原始噪点
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.4)
# 画出模型的拟合曲线
plt.plot(X_test_numpy, y_test_pred, label='Neural Network Fit', color='red', linewidth=3)

plt.title("Non-linear Regression with Neural Network")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()