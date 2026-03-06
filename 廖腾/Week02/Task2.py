import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 生成模拟数据
x = torch.linspace(0, 4 * torch.pi, 200).unsqueeze(1)
y_true = 2.5 * torch.sin(x + 0.7) + 0.5
y = y_true + 0.1 * torch.randn_like(y_true)


class NeuralSinFitter(nn.Module):
    """使用神经网络拟合正弦函数"""

    def __init__(self, hidden_size=64):
        super().__init__()
        # 3层神经网络
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),  # 第1层
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # 第2层
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # 第3层
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # 输出层
        )

    def forward(self, x):
        return self.net(x)


# 创建模型
model = NeuralSinFitter()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

# 训练
num_epochs = 1000
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())
    # 每100轮打印一次
    if (epoch + 1) % 30 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 训练后打印最终参数
print("\n" + "=" * 50)
print("训练完成！最终参数:")
print("=" * 50)

# 重新计算预测值（训练结束后）
with torch.no_grad():
    y_pred_final = model(x)
    final_loss = criterion(y_pred_final, y_true)
# 转换为numpy用于绘图
x_np = x.numpy().flatten()
y_true_np = y_true.numpy().flatten()
y_noisy_np = y.numpy().flatten()
y_pred_np = y_pred_final.numpy().flatten()

# 只绘制拟合效果图
plt.figure(figsize=(8, 5))
plt.scatter(x_np, y.numpy(), s=10, alpha=0.5, label='数据点', color='gray')
plt.plot(x_np, y_true_np, 'g-', linewidth=2, label='真实函数')
plt.plot(x_np, y_pred_np, 'r--', linewidth=2, label='神经网络拟合')
plt.xlabel('x')
plt.ylabel('y')
plt.title('正弦函数神经网络拟合')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()