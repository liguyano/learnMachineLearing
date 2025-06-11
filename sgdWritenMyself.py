import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# 使用你之前定义的数据
x_train = torch.linspace(-10, 10, 100).reshape(-1, 1)
y_train = 2 * x_train + 3 + torch.randn(x_train.size())


# 损失函数：均方误差（MSE）
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


# 初始化最小损失和对应参数
min_loss = float('inf')
best_k = 0
best_b = 0

# 设置搜索范围
k_values = np.arange(-10, 10, 0.1)  # 步长可以根据需要调整
b_values = np.arange(-10, 10, 0.1)

# 随机抽取的样本数量
num_samples = 10

# 开始暴力搜索
for k in k_values:
    for b in b_values:
        # 随机选取 num_samples 个样本的索引
        sample_indices = random.sample(range(len(x_train)), num_samples)

        # 根据索引获取相应的 x_train 和 y_train
        x_sample = x_train[sample_indices]
        y_sample = y_train[sample_indices]

        # 用当前 k 和 b 进行预测
        y_pred = k * x_sample + b

        # 计算损失
        loss = mse_loss(y_pred, y_sample).item()

        # 更新最优参数
        if loss < min_loss:
            min_loss = loss
            best_k = k
            best_b = b

# 输出结果
print(f"最佳 k (斜率): {best_k:.2f}")
print(f"最佳 b (偏置): {best_b:.2f}")
print(f"最小损失: {min_loss:.4f}")

# 绘制结果
plt.figure(figsize=(8, 6))

# 原始数据点
plt.scatter(x_train.numpy(), y_train.numpy(), color='blue', label='Original data')

# 真实线性关系
plt.plot(x_train.numpy(), (2 * x_train + 3).numpy(), color='green', label='True line (y = 2x + 3)')

# 最佳拟合直线
plt.plot(x_train.numpy(), (best_k * x_train + best_b).numpy(), color='red',
         label=f'Best fit line (y = {best_k:.2f}x + {best_b:.2f})')

plt.title('Linear Regression with Best Fit Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()