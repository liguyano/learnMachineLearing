from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 加载完整的鸢尾花数据集（包含所有三个类别）
data = load_iris()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树模型，设置更多参数
dt_model = DecisionTreeClassifier(
    criterion='gini',           # 分割标准：'gini' 或 'entropy'
    max_depth=10,              # 最大深度
    min_samples_split=2,       # 内部节点再划分所需最小样本数
    min_samples_leaf=1,        # 叶子节点最少样本数
    max_features=None,         # 寻找最佳分割时考虑的特征数量
    max_leaf_nodes=None,       # 最大叶子节点数
    min_impurity_decrease=0.0, # 节点划分最小不纯度减少
    random_state=42
)

# 训练模型
dt_model.fit(X_train, y_train)

# 预测
predictions = dt_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print(f"Decision Tree Accuracy: {accuracy:.4f}")

# 详细分类报告
print("\n分类报告:")
print(classification_report(y_test, predictions, target_names=data.target_names))

# 计算混淆矩阵
cm = confusion_matrix(y_test, predictions)

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.title('Decision Tree Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 可视化决策树结构
plt.figure(figsize=(20, 12))
plot_tree(dt_model, 
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()

# 特征重要性
feature_importance = dt_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(data.feature_names, feature_importance)
plt.title('Feature Importance in Decision Tree')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()

print("\n特征重要性:")
for i, importance in enumerate(feature_importance):
    print(f"{data.feature_names[i]}: {importance:.4f}")

# 网格搜索进行参数优化
print("\n进行网格搜索参数优化...")
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳参数训练模型
best_dt = grid_search.best_estimator_
best_predictions = best_dt.predict(X_test)
best_accuracy = accuracy_score(y_test, best_predictions)

print(f"优化后的决策树准确率: {best_accuracy:.4f}")

# 比较不同深度的性能
depths = range(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_temp.fit(X_train, y_train)
    
    train_score = dt_temp.score(X_train, y_train)
    test_score = dt_temp.score(X_test, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)

# 绘制深度vs性能图
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'o-', label='Training Accuracy', color='blue')
plt.plot(depths, test_scores, 'o-', label='Testing Accuracy', color='red')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Performance vs Max Depth')
plt.legend()
plt.grid(True)
plt.show()

print(f"\n决策树信息:")
print(f"树的深度: {dt_model.get_depth()}")
print(f"叶子节点数: {dt_model.get_n_leaves()}")
print(f"节点总数: {dt_model.tree_.node_count}") 