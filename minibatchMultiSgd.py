import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# 设置随机种子以保证可重复性
torch.manual_seed(42)

# ========================
# Step 1: 生成数据
# ========================
# 真实参数（我们希望通过训练恢复这些值）
true_k = 2.3
true_j = -1.5
true_b = 3.0

# 生成一些随机输入数据 x 和 y
num_samples = 1000
x = torch.randn(num_samples, 1)
y = torch.randn(num_samples, 1)

# 生成标签数据 z，并加入一些噪声
z = true_k * x + true_j * y + true_b
z =z + torch.randn(z.size()) * 0.5

# ========================
# Step 2: 初始化模型参数
# ========================
k = torch.randn(1, requires_grad=False)  # 权重 x 对应的系数
j = torch.randn(1, requires_grad=False)  # 权重 y 对应的系数
b = torch.randn(1, requires_grad=False)  # 偏置项

learning_rate = 0.01
num_epochs = 100
batch_size=10
# ========================
# Step 3: 开始训练 (SGD)
# ========================
print(f"k:{k}, j:{j}, b:{b}")
for epoch in range(num_epochs):
    indices = list(range(len(x)))
    random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]

        # 根据索引获取相应的 x_train 和 y_train
        x_batch = x[batch_indices]
        y_batch = y[batch_indices]
        z_batch = z[batch_indices]
        z_pred = k*x_batch+j*y_batch+b
        loss=((z_pred-z_batch)**2).mean()

        dk= (2*(z_pred-z_batch)*x_batch).mean()
        dj= (2*(z_pred-z_batch)*y_batch).mean()
        db= (2*(z_pred-z_batch)).mean()
        # 更新参数
        k -= learning_rate * dk
        j -= learning_rate * dj
        b -= learning_rate * db
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            print(f"k = {k.item():.2f}, j = {j.item():.2f}, b = {b.item():.2f}")


print("\n训练完成！")
print(f"真实参数: k={true_k}, j={true_j}, b={true_b}")
print(f"学习到的参数: k={k.item():.2f}, j={j.item():.2f}, b={b.item():.2f}")


# 创建网格数据用于绘图
x_plot = np.linspace(-3, 3, 100)
y_plot = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_plot, y_plot)
Z_true = true_k * X + true_j * Y + true_b
Z_pred = k.item() * X + j.item() * Y + b.item()

# 绘制三维图表
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据点
ax.scatter(x.numpy().flatten(), y.numpy().flatten(), z.numpy().flatten(), c='r', marker='o', label="Data Points")

# 绘制真实平面
ax.plot_surface(X, Y, Z_true, color='blue', alpha=0.5, label="True Plane")

# 绘制拟合平面
ax.plot_surface(X, Y, Z_pred, color='green', alpha=0.5, label="Fitted Plane")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("Linear Regression with Two Inputs: True vs Fitted Plane")
plt.legend()
plt.show()