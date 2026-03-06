"""
2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

# 生成sin函数数据
X_numpy = np.linspace(0, 4 * np.pi, 100).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) / 20
#2 * X_numpy + 1 + np.random.randn(100, 1)

# 将numpy数组转换为torch张量并设置为浮点类型
X = torch.from_numpy(X_numpy).float()
y= torch.from_numpy(y_numpy).float()

print("数据生成完成")
print("---" * 10)

# 定义多层网络
class MultiLayerNet(torch.nn.Module):
    def __init__(self):
        super(MultiLayerNet, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.activation = torch.nn.ReLU()
        self.output = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

# 初始化模型参数a（斜率）和b（截距），设置requires_grad=True以启用梯度计算
# a = torch.randn(1, requires_grad = True, dtype = torch.float)
# b = torch.randn(1, requires_grad = True, dtype = torch.float)

# print(f"初始参数 a：{a.item():.4f}")
# print(f"初始参数 b：{b.item():.4f}")
# print("---" * 10)

model = MultiLayerNet()
print("多层网络已创建")
print("---" * 10)

# 定义均方误差损失函数
loss_fn = torch.nn.MSELoss()

# 使用随机梯度下降优化器，学习率为0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 设置训练轮数
num_epochs = 10000
for epoch in range(num_epochs):
    # 前向传播：计算预测值
    # y_pred = a * X + b
    y_pred = model(X)

    # 计算预测值与真实值之间的损失
    loss = loss_fn(y_pred, y)

    # 清零梯度、反向传播计算梯度、更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss：{loss.item():.4f}")

print("\n训练完成")
# a_learned = a.item()
# b_learned = b.item()
# print(f"拟合的斜率 a：{a_learned:.4f}")
# print(f"拟合的截距 b：{b_learned:.4f}")
print(f"最终损失：{loss.item():.4f}")
print("---" * 10)

# 在推理阶段禁用梯度计算
with torch.no_grad():
    # y_predicted = a_learned * X + b_learned
    y_predicted = model(X)

# 绘制原始数据点和拟合直线
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label="sin(x) data", color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label="Fitted curve by neural network", color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Multi-layer Network Fitting of Sin Function')
plt.legend()
plt.grid(True)
plt.show()
