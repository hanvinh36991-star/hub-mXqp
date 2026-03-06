import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据：sin 函数 + 噪声
noise_scale = 0.1
X_numpy = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1) # 生成300个点，范围是-π到π
y_numpy = np.sin(X_numpy) + noise_scale * np.random.randn(300, 1)
X = torch.tensor(X_numpy, dtype=torch.float32)
y = torch.tensor(y_numpy, dtype=torch.float32)

# 2. 模型
# 仅对这个问题来说，我认为一个Tanh激活的隐藏层就已经足够好了，再多加几层反而容易受到噪声的影响（过拟合）
# 事实上也是如此，我多次运行发现，（1）的loss普遍比（2）的loss略小
# (1)：1 隐藏层（10 神经元）+ Tanh 激活
# model = nn.Sequential(
#     nn.Linear(1, 10),  # 全连接
#     nn.Tanh(),  # 非线性激活
#     nn.Linear(10, 1)  # 输出层（无激活）
# )

# (2)：3 个隐藏层
model = nn.Sequential(
    nn.Linear(1, 20),      # 全连接 + Tanh 激活，平滑处理
    nn.Tanh(),
    nn.Linear(20, 15),     # 全连接 + ELU 激活，负值非零处理，兼顾平滑与偏移修正
    nn.ELU(),
    nn.Linear(15, 10),     # 全连接 + SiLU 激活，负责可微的自门控，既保持平滑又能线性通过正域，细节不会丢
    nn.SiLU(),
    nn.Linear(10, 1)       # 输出层（无激活）
)

# 3. 损失与优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 1000  # 训练迭代次数
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}, loss={loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！各层学到的参数：")
linear_cnt = 0
for idx, m in enumerate(model):
    if isinstance(m, nn.Linear):          # 只关心 Linear 层
        linear_cnt += 1
        w = m.weight.data.numpy().flatten()
        b = m.bias.data.numpy().flatten()
        layer_name = "输出层" if idx == len(model) - 1 else f"隐藏层-{linear_cnt}"
        print(f"{layer_name}  weight范数: {np.linalg.norm(w):.4f}  bias范数: {np.linalg.norm(b):.4f}")

# 6. 将模型切换到评估模式
model.eval()

# 7. 禁用梯度计算，因为不再训练
X_plot = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1) # 1000 点，用比源数据点更密集的数据进行预测
X_plot_tensor = torch.tensor(X_plot, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_plot_tensor).numpy() # 使用训练好的模型进行预测

# 8. 画图
plt.scatter(X_numpy, y_numpy, s=10, label='Noisy sin')
plt.plot(X_plot, y_pred, 'r', lw=2, label='MLP fit')  # 密网格→光滑曲线
plt.legend()
plt.show()
