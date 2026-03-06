import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# 1. 生成Sin函数模拟数据（带噪声，更贴近真实场景）
np.random.seed(42)  # 固定随机种子，结果可复现
X_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)  # 0到2π生成200个点，形状(200,1)
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)  # sin(x) + 小噪声

# 转换为PyTorch张量（float类型，用于网络计算）
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("Sin函数数据生成完成（带噪声）")
print("数据形状：X={}, y={}".format(X.shape, y.shape))
print("---" * 15)

# 2. 定义多层全连接神经网络（MLP），用于拟合非线性Sin函数
# 输入层(1维) → 隐藏层1(32维) → 隐藏层2(64维) → 隐藏层3(32维) → 输出层(1维)
# 采用ReLU激活函数实现非线性映射，适配Sin函数的非线性特征
class SinFittingNet(nn.Module):
    def __init__(self):
        super(SinFittingNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=1, out_features=32),  # 输入层→隐藏层1
            nn.ReLU(),  # 非线性激活，关键：无激活则退化为线性模型，无法拟合Sin
            nn.Linear(32, 64),  # 隐藏层1→隐藏层2
            nn.ReLU(),
            nn.Linear(64, 32),  # 隐藏层2→隐藏层3
            nn.ReLU(),
            nn.Linear(32, 1)   # 隐藏层3→输出层（回归任务，无激活）
        )
    
    # 前向传播：定义数据通过网络的计算路径
    def forward(self, x):
        return self.layers(x)

# 初始化网络、损失函数、优化器
model = SinFittingNet()  # 实例化多层网络
loss_fn = nn.MSELoss()   # 回归任务仍用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，比SGD收敛更快更稳定
print("多层网络初始化完成：")
print(model)  # 打印网络结构
print("---" * 15)

# 3. 训练网络拟合Sin函数
num_epochs = 3000  # 训练轮数，适配非线性拟合需求
loss_history = []  # 记录每轮损失，用于后续可视化

for epoch in range(num_epochs):
    # 前向传播：输入X，得到网络预测值y_pred
    y_pred = model(X)
    
    # 计算损失：预测值与真实值的均方误差
    loss = loss_fn(y_pred, y)
    loss_history.append(loss.item())  # 保存损失值
    
    # 反向传播+优化：梯度清零→计算梯度→更新参数
    optimizer.zero_grad()  # 清空上一轮梯度（PyTorch梯度默认累加）
    loss.backward()        # 反向传播，计算所有可训练参数的梯度
    optimizer.step()       # 根据梯度更新网络参数
    
    # 每300轮打印一次训练状态，观察损失下降
    if (epoch + 1) % 300 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

print("---" * 15)
print("Sin函数拟合训练完成！")

# 4. 可视化：①训练损失变化 ②网络拟合结果与真实Sin函数对比
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 1行2列的子图

# 子图1：训练损失曲线（观察网络收敛情况）
ax1.plot(range(num_epochs), loss_history, color='#FF6347', linewidth=2)
ax1.set_xlabel('训练轮数 (Epoch)', fontsize=12)
ax1.set_ylabel('均方误差损失 (MSE Loss)', fontsize=12)
ax1.set_title('多层网络训练损失变化曲线', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, num_epochs)

# 子图2：拟合结果与真实Sin函数对比（核心可视化）
with torch.no_grad():  # 推理阶段，关闭梯度计算，节省资源
    y_pred = model(X).numpy()  # 网络预测值转换为numpy数组，用于绘图

# 绘制原始带噪声数据、真实Sin曲线、网络拟合曲线
ax2.scatter(X_numpy, y_numpy, color='#1E90FF', alpha=0.6, s=15, label='带噪声原始数据')
ax2.plot(X_numpy, np.sin(X_numpy), color='#228B22', linewidth=3, label='真实sin(x)曲线')
ax2.plot(X_numpy, y_pred, color='#FF4500', linewidth=3, linestyle='--', label='网络拟合曲线')
ax2.set_xlabel('x (0 ~ 2π)', fontsize=12)
ax2.set_ylabel('sin(x)', fontsize=12)
ax2.set_title('多层网络拟合sin(x)结果对比', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 2*np.pi)

plt.tight_layout()  # 调整子图间距，避免重叠
plt.show()

# 5. （可选）测试网络泛化能力：用新的x值测试拟合效果
# test_x = torch.linspace(0, 2*np.pi, 100).reshape(-1,1).float()
# with torch.no_grad():
#     test_y = model(test_x)
# plt.plot(test_x.numpy(), test_y.numpy(), color='purple', linewidth=2, label='泛化测试曲线')
