from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 生成一些简单数据
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.squeeze() + 1 + np.random.randn(100)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print("MSE:", mean_squared_error(y_test, y_pred))
print("系数 a:", model.coef_)
print("截距 b:", model.intercept_)