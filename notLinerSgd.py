import numpy as np
import matplotlib
# 设置随机种子（可选）
np.random.seed(0)

# =================== Step 1: 构造数据 ===================
# 真实参数 a=2, b=0.5
x_data = np.linspace(0, 4, 100)
y_true = 2 * np.exp(0.5 * x_data)
y_data = y_true + np.random.normal(size=len(x_data)) * 1  # 添加噪声


# =================== Step 2: 定义模型 ===================
def model(x, a, b):
    return a * np.exp(b * x)


# =================== Step 3: 损失函数 MSE ===================
def compute_loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)


# =================== Step 4: 参数初始化 ===================
a = 1.0  # 初始值
b = 0.1  # 初始值
learning_rate = 0.001
epochs = 10000
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False     # 解决负号 '-' 显示为方块的问题
# =================== Step 5: 训练过程 ===================
for epoch in range(epochs):
    y_pred = model(x_data, a, b)

    # 计算梯度
    error = y_pred - y_data
    grad_a = (2 / len(x_data)) * np.sum(error * np.exp(b * x_data))
    grad_b = (2 / len(x_data)) * np.sum(error * a * x_data * np.exp(b * x_data))

    # 参数更新
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b

    if epoch % 1000 == 0:
        loss = compute_loss(y_data, y_pred)
        print(f"Epoch {epoch}: Loss = {loss:.6f}, a = {a:.4f}, b = {b:.4f}")

# =================== Step 6: 最终结果 ===================
print("\n最终拟合参数:")
print(f"a = {a:.4f}, b = {b:.4f}")
print("真实参数: a=2.0, b=0.5")

# 可视化
import matplotlib.pyplot as plt

plt.scatter(x_data, y_data, label='原始数据')
plt.plot(x_data, model(x_data, a, b), 'r', label='拟合曲线')
plt.xlabel('x')
plt.ylabel('y')
plt.title('非线性回归手动实现 (SGD)')
plt.legend()
plt.show()