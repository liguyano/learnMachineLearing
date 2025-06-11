import torch
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
torch.manual_seed(42)

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
num_epochs = 500

# ========================
# Step 3: 手动实现 SGD
# ========================
for epoch in range(num_epochs):
    # 随机选择一个样本（模拟 SGD，每次只用一个样本）
    idx = torch.randint(0, len(x_train), (1,))
    x = x_train[idx]
    y_true = y_train[idx]

    # 前向传播：计算预测值
    y_pred = k * x + b

    # 计算损失（MSE）
    loss = (y_pred - y_true) ** 2

    # 手动计算梯度
    # 因为不使用自动求导，我们手动写出梯度公式：
    # dloss/dk = 2*(y_pred - y_true)*x
    # dloss/db = 2*(y_pred - y_true)
    dk = 2 * (y_pred - y_true) * x.item()
    db = 2 * (y_pred - y_true)

    # 更新参数
    k = k - learning_rate * dk
    b = b - learning_rate * db

    # 打印训练信息
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, k={k.item():.2f}, b={b.item():.2f}")

# ========================
# Step 4: 绘图显示结果
# ========================
plt.scatter(x_train.numpy(), y_train.numpy(), label='Original data')
plt.plot(x_train.numpy(), (2 * x_train + 3).numpy(), 'g', label='True line (y=2x+3)')
plt.plot(x_train.numpy(), (k * x_train + b).detach().numpy(), 'r', label=f'Fitted line (y={k.item():.2f}x + {b.item():.2f})')
plt.legend()
plt.title("Manual SGD Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()