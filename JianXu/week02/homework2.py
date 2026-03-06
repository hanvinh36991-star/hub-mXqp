import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import torch.nn as nn


# 2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 10
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(100, 1)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)
print(X)
print(y)
print(X.shape, y.shape) # x.shape (100, 1) y.shape (100, 1)



# 3. 定义学习率
learning_rate = 0.005

# 4. 训练模型
num_epochs = 5000

# 5. 网络层
model = nn.Sequential(
    nn.Linear(1, 64), # (B, 1) -> (B, 64)
    nn.Tanh(),
    nn.Linear(64, 64),  # (B, 64) -> (B, 64)
    nn.Tanh(),
    nn.Linear(64, 1)  # (B, 64) -> (B, 1)
)

# 6. 损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 7. 训练模型
for epoch in range(num_epochs):
    model.train()

    y_pred = model(X)
    # 计算 MSE 损失
    loss = loss_fn(y_pred, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
print("---" * 10)

# 8 可视化：原始数据 scatter + 真值曲线 + 预测曲线
model.eval()
with torch.no_grad():
    X_plot = np.linspace(0, 10, 500).reshape(-1, 1).astype(np.float32)
    y_true = np.sin(X_plot)

    X_plot_t = torch.from_numpy(X_plot)
    y_plot_pred = model(X_plot_t).numpy()

plt.figure(figsize=(10, 6))

# 原始数据
plt.scatter(X_numpy, y_numpy, label="Noisy data (scatter)", alpha=0.5)
# 真值 sin：用曲线
plt.plot(X_plot, y_true, label="True sin(x)", linewidth=2)
# 模型预测：用曲线
plt.plot(X_plot, y_plot_pred, label="DL prediction", linewidth=2)

plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()