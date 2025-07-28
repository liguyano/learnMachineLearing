from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 加载数据集
data = load_iris()
X = data.data[data.target != 2]  # 只选择前两类
y = data.target[data.target != 2]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测
predictions = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

# 计算混淆矩阵
cm = confusion_matrix(y_test, predictions)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 打印混淆矩阵的详细解释
print("\n混淆矩阵解释:")
print("真阳性 (True Positive):", cm[1][1])
print("假阳性 (False Positive):", cm[0][1])
print("真阴性 (True Negative):", cm[0][0])
print("假阴性 (False Negative):", cm[1][0]) 