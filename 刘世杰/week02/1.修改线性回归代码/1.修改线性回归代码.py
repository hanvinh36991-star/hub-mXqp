import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统的黑体，Mac用['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题

# 1. 生成sin函数数据（带噪声）
# 生成0到2π之间的1000个点，增加数据量提升拟合效果
X_numpy = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)  # (1000, 1)
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)  # 加噪声的sin曲线

# 转换为torch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成，数据形状：X={}, y={}".format(X.shape, y.shape))
print("---" * 10)


# 2. 定义多层神经网络（替代原线性模型）
class SinFittingNet(nn.Module):
    def __init__(self):
        super(SinFittingNet, self).__init__()
        # 多层网络：输入层(1) -> 隐藏层1(64) -> 隐藏层2(32) -> 输出层(1)
        self.layers = nn.Sequential(
            nn.Linear(1, 64),  # 输入层：1个特征（x）→ 64个隐藏节点
            nn.ReLU(),  # 激活函数（非线性）
            nn.Linear(64, 32),  # 隐藏层1 → 隐藏层2
            nn.ReLU(),  # 激活函数
            nn.Linear(32, 1)  # 输出层：32 → 1个输出（y）
        )

    def forward(self, x):
        return self.layers(x)


# 3. 初始化模型、损失函数、优化器
model = SinFittingNet()
loss_fn = nn.MSELoss()  # 回归任务仍用MSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器效果更好

# 4. 训练模型
num_epochs = 5000
loss_records = []  # 记录Loss变化
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)
    loss_records.append(loss.item())

    # 反向传播+优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每500个epoch打印一次
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 可视化训练过程的Loss
plt.figure(figsize=(10, 4))
plt.plot(range(1, num_epochs + 1), loss_records)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("训练过程Loss变化")
plt.grid(True)
plt.show()

# 6. 预测并可视化拟合效果
model.eval()  # 切换到评估模式
with torch.no_grad():
    y_pred = model(X).numpy()  # 预测结果转换为numpy

# 绘制原始数据和拟合曲线
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='带噪声的sin数据', color='blue', alpha=0.3, s=5)
plt.plot(X_numpy, y_pred, label='多层网络拟合曲线', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), label='纯sin函数（无噪声）', color='green', linestyle='--', linewidth=2)
plt.xlabel('X (0 ~ 2π)')
plt.ylabel('y = sin(X) + 噪声')
plt.legend()
plt.grid(True)
plt.title('多层神经网络拟合sin函数')
plt.show()

# 7. 测试泛化能力
X_test = np.linspace(2 * np.pi, 4 * np.pi, 500).reshape(-1, 1)  # 2π~4π的点（训练集未覆盖）
X_test_tensor = torch.from_numpy(X_test).float()
with torch.no_grad():
    y_test_pred = model(X_test_tensor).numpy()

plt.figure(figsize=(10, 6))
plt.plot(X_test, np.sin(X_test), label='真实sin函数（2π~4π）', color='green', linewidth=2)
plt.plot(X_test, y_test_pred, label='模型预测（2π~4π）', color='red', linestyle='--', linewidth=2)
plt.xlabel('X (2π ~ 4π)')
plt.ylabel('y = sin(X)')
plt.legend()
plt.grid(True)
plt.title('模型泛化能力测试（超出训练范围）')
plt.show()
