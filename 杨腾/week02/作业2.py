import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 固定随机数生成器的状态，保证模型初始化和训练过程的可重复性
np.random.seed(42)
torch.manual_seed(42)

# 2. 生成 sin 数据
X_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
y_numpy = np.sin(X_numpy)

# 转为 torch Tensor
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin 数据生成完成")
print("-" * 30)

# 3. 定义多层神经网络
class SinNet(torch.nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 1)

        # 对拟合连续函数，添加激活函数
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

model = SinNet()
print(model)
print("-" * 30)

# 4. 损失函数 & 优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # 测试验证优化器 Adam 比 SGD 更稳定，效果更好

# 5. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("\n训练完成")
print("-" * 30)

# 6. 推理 & 可视化
model.eval()
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label="True sin(x)", color="blue", alpha=0.6)
plt.plot(X_numpy, y_predicted, label="Model Prediction", color="red", linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
