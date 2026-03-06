import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 1. 生成模拟数据
# 生成 -5 到 5 之间的随机点
X_numpy = np.random.uniform(-5, 5, (500, 1))
# 目标函数 y = sin(x)，并加入少量噪声
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(500, 1)


X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()


print("数据生成完成：拟合目标为 sin(x)")
print("---" * 10)


# 2. 构建多层神经网络模型
# 使用 nn.Sequential 快速搭建包含隐藏层的网络
# 输入(1) -> 隐藏层1(64) -> 激活函数 -> 隐藏层2(64) -> 激活函数 -> 输出(1)
model = nn.Sequential(
    nn.Linear(1, 64),     # 输入层到隐藏层1
    nn.ReLU(),            # 激活函数，提供非线性能力
    nn.Linear(64, 64),    # 隐藏层1到隐藏层2
    nn.ReLU(),            # 激活函数
    nn.Linear(64, 1)      # 隐藏层2到输出层
)


# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss()
# 使用 Adam 优化器，它在处理非线性拟合时比 SGD 更高效
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每 200 个 epoch 打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 5. 准备可视化数据
# 生成一组连续的 X 用于绘制平滑的拟合曲线
X_test_np = np.linspace(-5, 5, 200).reshape(-1, 1)
X_test_torch = torch.from_numpy(X_test_np).float()


# 预测时不需要计算梯度
with torch.no_grad():
    y_test_pred = model(X_test_torch).numpy()


# 6. 绘制结果
plt.figure(figsize=(10, 6))
# 原始数据散点
plt.scatter(X_numpy, y_numpy, label='Raw data (sin + noise)', color='blue', alpha=0.3)
# 模型拟合的曲线
plt.plot(X_test_np, y_test_pred, label='Neural Network Fit', color='red', linewidth=3)
# 标准 sin 曲线（参考线）
plt.plot(X_test_np, np.sin(X_test_np), label='True sin(x)', color='green', linestyle='--', alpha=0.7)


plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()