import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

X_numpy = np.linspace(0, 4*np.pi, 1000).reshape(-1, 1)  # 生成0到4π范围内的数据点
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # sin函数加上少量噪声
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 定义多层神经网络
class SinNet(torch.nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 64)  # 输入层到第一隐藏层
        self.hidden2 = torch.nn.Linear(64, 32)  # 第一隐藏层到第二隐藏层
        self.hidden3 = torch.nn.Linear(32, 16)  # 第二隐藏层到第三隐藏层
        self.output = torch.nn.Linear(16, 1)    # 第三隐藏层到输出层
        self.activation = torch.nn.ReLU()       # 激活函数

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x = self.output(x)
        return x

# 创建网络实例
net = SinNet()

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务

optimizer = torch.optim.Adam(net.parameters(), lr=0.01) 

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    y_pred = net(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f"最终损失: {loss.item():.4f}")
with torch.no_grad():
    X_test = torch.linspace(0, 4*np.pi, 1000).reshape(-1, 1).float()
    y_predicted = net(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_test.numpy(), y_predicted.numpy(), label='Fitted curve', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), label='True sin(x)', color='green', linestyle='--')  # 显示真实sin函数
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
