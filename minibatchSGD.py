import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# ========================
# Step 1: 准备数据
# ========================
# 创建训练数据
x_train = torch.linspace(-10, 10, 100).reshape(-1, 1)
y_train = 2 * x_train + 3 + torch.randn(x_train.size()) * 0.5  # 添加少量噪声

# ========================
# Step 2: 初始化参数
# ========================
# 我们手动定义 k 和 b，并设置 requires_grad=False，因为我们要自己更新它们
k = torch.randn(1, requires_grad=False)  # 斜率
b = torch.randn(1, requires_grad=False)  # 偏置

# 学习率
learning_rate = 0.01

# 训练轮数
num_epochs = 100

# 小批量大小
batch_size = 10

# ========================
# Step 3: 手动实现 Mini-batch SGD
# ========================
for epoch in range(num_epochs):
    # 随机打乱数据索引
    indices = list(range(len(x_train)))
    random.shuffle(indices)

    # 分割成小批量
    for i in range(0, len(indices), batch_size):
        # 获取当前小批量的索引
        batch_indices = indices[i:i + batch_size]

        # 根据索引获取相应的 x_train 和 y_train
        x_batch = x_train[batch_indices]
        y_batch = y_train[batch_indices]

        # 前向传播：计算预测值
        y_pred = k * x_batch + b

        # 计算损失（MSE）
        loss = ((y_pred - y_batch) ** 2).mean()

        # 手动计算梯度
        dk = (2 * (y_pred - y_batch) * x_batch).mean()
        db = (2 * (y_pred - y_batch)).mean()

        # 更新参数
        k -= learning_rate * dk
        b -= learning_rate * db

    # 每隔一定轮次打印一次信息
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, k={k.item():.2f}, b={b.item():.2f}")

# ========================
# Step 4: 绘图显示结果
# ========================
plt.scatter(x_train.numpy(), y_train.numpy(), label='Original data')
plt.plot(x_train.numpy(), (2 * x_train + 3).numpy(), 'g', label='True line (y=2x+3)')
plt.plot(x_train.numpy(), (k * x_train + b).detach().numpy(), 'r',
         label=f'Fitted line (y={k.item():.2f}x + {b.item():.2f})')
plt.legend()
plt.title("Manual Mini-batch SGD Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()