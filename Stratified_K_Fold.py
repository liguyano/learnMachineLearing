from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
from sklearn.svm import SVC

# 加载示例数据集
digits = datasets.load_digits()

# 获取特征和标签
X = digits.data
y = digits.target

# 创建分层K折交叉验证对象
skf = StratifiedKFold(n_splits=5)

# 初始化SVC分类器
clf = SVC(kernel='linear')

# 计算每折的准确率
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"本折准确率: {score:.2f}")