"""
2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
"""

# 1.构建训练集
# 2.构建一个多层网络模型
# 3.拟合、训练
# 4.可视化
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn


# 1. 生成模拟数据
def generate_sin_data(num_samples=1000, noise_level=0.05):
    """
    生成sin函数数据，添加高斯噪声

    参数:
    num_samples: 样本数量
    noise_level: 噪声水平

    返回:
    x_train, y_train: 训练数据
    x_test, y_test: 测试数据
    """
    # 训练数据（带噪声）
    x_train = torch.rand(num_samples, 1) * 4 * np.pi - 2 * np.pi  # [-2π, 2π]
    y_true = torch.sin(2 * x_train)
    noise = torch.randn(num_samples, 1) * noise_level
    y_train = y_true + noise

    # 测试数据（无噪声，用于评估）
    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)
    y_test = torch.sin(2 * x_test)

    return x_train, y_train, x_test, y_test, y_true


# 生成数据
x_train, y_train, x_test, y_test, y_true = generate_sin_data(num_samples=500, noise_level=0.1)

print(f"训练数据: x_train shape={x_train.shape}, y_train shape={y_train.shape}")
print(f"测试数据: x_test shape={x_test.shape}, y_test shape={y_test.shape}")


a = torch.randn(1, requires_grad=True, dtype=torch.float)
print(f"初始参数 a: {a.item():.4f}")


# 2.构建一个多层网络模型
class SimpleClassifier(nn.Module):
    """
    创建模型
    """
    def __init__(self, input_dim, hidden_dim, output_dim):  # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.el = nn.ELU()
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.el(out)
        out = self.fc4(out)
        return out


input_dim = 1
hide_dim = 128
output_dim = 1
# 实例化自定义深度学习模型
model = SimpleClassifier(input_dim, hide_dim, output_dim)
# 定义损失函数
nnLoss = nn.MSELoss()
# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3.拟合、训练
num_epochs = 10000  # 训练迭代次数
for epoch in range(num_epochs):
    model.train()
    y_pre = model(x_train)
    loss = nnLoss(y_pre, y_train)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()

# 4. 绘制结果
with torch.no_grad():
    y_predicted = model(x_train).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, label='Raw data', color='blue', alpha=0.6)
plt.plot(x_train, y_predicted, label=f'Model: y = sin(x)', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
