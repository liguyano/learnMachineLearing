import torch
import torch.nn as nn
import torch.optim as optim

# 设置参数
input_size = 1
output_size = 1
num_epochs = 100
learning_rate = 0.01

# 创建一些假数据
# 真实权重为2，偏置为3
x_train = torch.linspace(-10, 10, 100).reshape(-1, 1)
y_train = 2 * x_train + 3 + torch.randn(x_train.size())  # 添加一些噪声


# 定义模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegressionModel(input_size, output_size)

# 损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 使用SGD优化器

# 训练模型
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
predicted = model(x_train).detach()
print('真实权重：', 2)
print('预测权重：', model.linear.weight.item())
print('真实偏置：', 3)
print('预测偏置：', model.linear.bias.item())

# 绘制结果
import matplotlib.pyplot as plt

plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predicted.numpy(), label='Fitted line')
plt.legend()
plt.show()