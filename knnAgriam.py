# 导入所需库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 加载数据集
iris = load_iris()
X = iris.data  # 特征 (4个特征)
y = iris.target  # 标签 (3个类别)

# 2. 划分训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. 特征标准化（建议，提升KNN效果）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 创建 KNN 分类器（K=5）
knn = KNeighborsClassifier(n_neighbors=5)

# 5. 训练模型（KNN 其实只是存储了训练数据）
knn.fit(X_train, y_train)

# 6. 在测试集上进行预测
y_pred = knn.predict(X_test)

# 7. 评估模型性能
print("准确率 Accuracy:", accuracy_score(y_test, y_pred))
print("\n分类报告 Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("混淆矩阵 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))