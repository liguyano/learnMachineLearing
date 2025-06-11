from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.svm import SVC

# 加载示例数据集
digits = datasets.load_digits()

# 获取特征和标签
X = digits.data
y = digits.target

# 创建分类器
clf = SVC(kernel='linear')

# 执行5折交叉验证
scores = cross_val_score(clf, X, y, cv=5)

print(f"每折准确率: {scores}")
print(f"平均准确率: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")