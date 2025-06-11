from sklearn.model_selection import train_test_split
from sklearn import datasets

# 加载示例数据集
digits = datasets.load_digits()

# 获取特征和标签
X = digits.data
y = digits.target

# 切分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")