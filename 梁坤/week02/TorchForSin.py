import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim


# 1. 生成模拟数据
# 使用linspace生成均匀分布的点，便于观察拟合效果
X_numpy = np.linspace(0, 10, 200).reshape(-1, 1)  # 生成0到10之间的200个均匀分布点
y_numpy = np.sin(X_numpy) + np.random.normal(0, 0.1, X_numpy.shape)  # 添加较小的噪声

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()


# 2. 定义多层感知机模型
class MLP(nn.Module):
	def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
		super(MLP, self).__init__()

		self.network = nn.Sequential(
			nn.Linear(input_size, hidden_size1),
			nn.ReLU(),
			nn.Linear(hidden_size1, hidden_size2),
			nn.ReLU(),
			nn.Linear(hidden_size2, output_size)
		)

	def forward(self, x):
		return self.network(x)


# 3. 初始化模型
input_size = 1
hidden_size1 = 32
hidden_size2 = 16
output_size = 1

model = MLP(input_size, hidden_size1, hidden_size2, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 2000
print("开始训练...")
for epoch in range(num_epochs):
	# 前向传播
	y_pred = model(X)

	# 计算损失
	loss = criterion(y_pred, y)

	# 反向传播
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	# 每500个epoch打印一次损失
	if (epoch + 1) % 500 == 0:
		print(f'Epoch [{epoch + 1:4d}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 使用训练好的模型进行预测
model.eval()
with torch.no_grad():
	y_predicted = model(X).numpy()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK']  # Mac中文
plt.rcParams['axes.unicode_minus'] = False

# 6. 可视化拟合结果
plt.figure(figsize=(12, 8))

# 创建单个图形
plt.scatter(X_numpy, y_numpy, label='训练数据 (sin(x) + 噪声)',
			color='blue', alpha=0.5, s=30, edgecolors='black', linewidths=0.5)

# 绘制模型的预测曲线
plt.plot(X_numpy, y_predicted, label='MLP预测结果',
		 color='red', linewidth=3, zorder=5)

# 绘制真实的sin函数曲线
plt.plot(X_numpy, np.sin(X_numpy), label='真实 sin(x) 函数',
		 color='green', linestyle='--', linewidth=2.5, alpha=0.8)

# 设置图形属性
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('多层感知机拟合 sin(x) 函数', fontsize=16, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')

# 设置坐标轴范围
plt.ylim(-1.5, 1.5)

# 添加背景网格
plt.grid(True, which='both', alpha=0.2)

# 在图上添加文本信息
plt.text(0.5, -1.3, f'最终损失: {loss.item():.6f}',
		 fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
plt.text(0.5, -1.4, f'网络结构: 1-{hidden_size1}-{hidden_size2}-1',
		 fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.show()

# 7. 打印最终结果
print("\n" + "=" * 60)
print("训练完成！")
print(f"最终损失: {loss.item():.6f}")
print(f"网络结构: 输入层(1) -> 隐藏层1({hidden_size1}) -> 隐藏层2({hidden_size2}) -> 输出层(1)")