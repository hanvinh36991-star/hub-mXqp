import torch
import numpy as np
import matplotlib.pyplot as plt

# ======================
# 1. 生成模拟数据（sin）
# ======================
X_numpy = np.random.rand(200, 1) * 2 * np.pi  # [0, 2π]
y_numpy = np.sin(X_numpy) + np.random.randn(200, 1) * 0.1  # 加一点噪声

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("-" * 30)

# ======================
# 2. 定义多层神经网络
# ======================
class SinNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SinNet()
print(model)
print("-" * 30)

# ======================
# 3. 损失函数 & 优化器
# ======================
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ======================
# 4. 训练模型
# ======================
num_epochs = 3000
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 300 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

print("\n训练完成！")
print("-" * 30)

# ======================
# 5. 可视化结果
# ======================
with torch.no_grad():
    X_test = torch.linspace(0, 2 * np.pi, 300).view(-1, 1)
    y_test_pred = model(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label="Raw data", alpha=0.4)
plt.plot(X_test.numpy(), np.sin(X_test.numpy()), label="True sin(x)", color="green")
plt.plot(X_test.numpy(), y_test_pred.numpy(), label="NN prediction", color="red", linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
