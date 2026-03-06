import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子以确保结果可重复
np.random.seed(42)
torch.manual_seed(42)


# 1. 生成sin函数数据
def generate_sin_data(n_samples=1000, noise_level=0.05):
    """生成sin函数数据，添加一些噪声"""
    X = np.random.uniform(-2 * np.pi, 2 * np.pi, n_samples).reshape(-1, 1)
    y = np.sin(X) + np.random.normal(0, noise_level, X.shape)
    return X, y


# 生成数据
X, y = generate_sin_data(1000, 0.05)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# 2. 定义神经网络模型
class SinApproximator(nn.Module):
    """多层神经网络用于拟合sin函数"""

    def __init__(self, hidden_layers=[64, 128, 64], activation='relu', dropout_rate=0.0):
        super(SinApproximator, self).__init__()

        layers = []
        input_size = 1

        # 构建隐藏层
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))

            # 添加激活函数
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            else:
                layers.append(nn.ReLU())

            # 添加dropout（可选）
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            input_size = hidden_size

        # 输出层
        layers.append(nn.Linear(input_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 3. 训练函数
def train_model(model, X_train, y_train, X_test, y_test,
                learning_rate=0.01, epochs=2000, batch_size=32):
    """训练神经网络模型"""

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 准备数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 记录训练过程
    train_losses = []
    test_losses = []

    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            # 前向传播
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 计算测试损失
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, y_test).item()
            test_losses.append(test_loss)

        # 每100个epoch打印一次进度
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}')

    return train_losses, test_losses


# 4. 创建并训练模型
print("创建神经网络模型...")
model = SinApproximator(hidden_layers=[64, 128, 64, 32], activation='tanh', dropout_rate=0.0)
print(f"模型结构:\n{model}")
print(f"总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

print("\n开始训练...")
train_losses, test_losses = train_model(
    model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
    learning_rate=0.005, epochs=1500, batch_size=64
)

# 5. 可视化结果
fig = plt.figure(figsize=(16, 10))

# 5.1 损失函数变化
plt.subplot(2, 3, 1)
plt.plot(train_losses, label='训练损失', linewidth=2)
plt.plot(test_losses, label='测试损失', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('训练和测试损失')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.2 原始数据与拟合结果对比
plt.subplot(2, 3, 2)
# 生成用于预测的x值（按顺序排列）
x_plot = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)
x_plot_tensor = torch.tensor(x_plot, dtype=torch.float32)

# 真实sin函数值
y_true = np.sin(x_plot)

# 模型预测值
with torch.no_grad():
    model.eval()
    y_pred = model(x_plot_tensor).numpy()

# 绘制原始数据点
plt.scatter(X_train, y_train, alpha=0.3, s=10, label='训练数据', color='blue')
plt.plot(x_plot, y_true, 'g-', linewidth=3, label='真实sin函数')
plt.plot(x_plot, y_pred, 'r-', linewidth=3, label='神经网络拟合')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('神经网络拟合sin函数')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.3 预测误差
plt.subplot(2, 3, 3)
error = y_pred.flatten() - y_true.flatten()
plt.plot(x_plot, error, 'b-', linewidth=2)
plt.fill_between(x_plot.flatten(), error, 0, alpha=0.3)
plt.xlabel('x')
plt.ylabel('预测误差')
plt.title('拟合误差')
plt.grid(True, alpha=0.3)

# 5.4 残差图
plt.subplot(2, 3, 4)
with torch.no_grad():
    y_train_pred = model(X_train_tensor).numpy()
    y_test_pred = model(X_test_tensor).numpy()

plt.scatter(y_train, y_train_pred, alpha=0.5, s=20, label='训练集', color='blue')
plt.scatter(y_test, y_test_pred, alpha=0.5, s=20, label='测试集', color='red')
plt.plot([-1.5, 1.5], [-1.5, 1.5], 'k--', linewidth=2, label='完美预测')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('预测值与真实值对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 5.5 模型在训练集和测试集上的表现
plt.subplot(2, 3, 5)
with torch.no_grad():
    # 计算训练集和测试集的MSE
    train_mse = np.mean((y_train_pred - y_train) ** 2)
    test_mse = np.mean((y_test_pred - y_test) ** 2)

    # 计算R²分数
    train_mean = np.mean(y_train)
    test_mean = np.mean(y_test)

    train_ss_total = np.sum((y_train - train_mean) ** 2)
    train_ss_residual = np.sum((y_train - y_train_pred) ** 2)
    train_r2 = 1 - (train_ss_residual / train_ss_total)

    test_ss_total = np.sum((y_test - test_mean) ** 2)
    test_ss_residual = np.sum((y_test - y_test_pred) ** 2)
    test_r2 = 1 - (test_ss_residual / test_ss_total)

metrics = ['训练集MSE', '测试集MSE', '训练集R²', '测试集R²']
values = [train_mse, test_mse, train_r2, test_r2]
colors = ['blue', 'red', 'blue', 'red']

bars = plt.bar(metrics, values, color=colors, alpha=0.7)
plt.ylabel('值')
plt.title('模型性能指标')
plt.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值标签
for bar, value in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{value:.4f}', ha='center', va='bottom')

# 5.6 不同隐藏层大小的对比
plt.subplot(2, 3, 6)
hidden_sizes = [[16], [32, 16], [64, 32, 16], [128, 64, 32, 16]]
colors = ['red', 'green', 'blue', 'purple']
labels = ['1层(16)', '2层(32,16)', '3层(64,32,16)', '4层(128,64,32,16)']

plt.plot(x_plot, y_true, 'k-', linewidth=3, label='真实sin函数')

for i, hidden_size in enumerate(hidden_sizes):
    # 创建并训练一个简单模型用于比较
    simple_model = SinApproximator(hidden_layers=hidden_size, activation='tanh')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(simple_model.parameters(), lr=0.01)

    # 快速训练
    for epoch in range(500):
        predictions = simple_model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 预测
    with torch.no_grad():
        simple_model.eval()
        y_simple_pred = simple_model(x_plot_tensor).numpy()

    plt.plot(x_plot, y_simple_pred, '--', linewidth=2, color=colors[i], label=labels[i])

plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('不同网络深度拟合效果对比')
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. 打印最终评估结果
print("\n" + "=" * 50)
print("最终模型评估:")
print(f"训练集MSE: {train_mse:.6f}")
print(f"测试集MSE: {test_mse:.6f}")
print(f"训练集R²分数: {train_r2:.6f}")
print(f"测试集R²分数: {test_r2:.6f}")
print("=" * 50)

# 7. 生成测试数据并显示模型预测
print("\n测试模型在特定点的预测:")
test_points = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]).reshape(-1, 1)
test_points_tensor = torch.tensor(test_points, dtype=torch.float32)  

with torch.no_grad():
    model.eval()
    predictions = model(test_points_tensor).numpy()

for i, (x, pred) in enumerate(zip(test_points.flatten(), predictions.flatten())):
    true_val = np.sin(x)
    error = abs(pred - true_val)
    print(f"x = {x:6.3f}, sin(x) = {true_val:6.3f}, 预测值 = {pred:6.3f}, 误差 = {error:6.3f}")
