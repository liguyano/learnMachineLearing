import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Node:
    """决策树节点类"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, samples=None):
        self.feature = feature      # 分割特征的索引
        self.threshold = threshold  # 分割阈值
        self.left = left           # 左子树
        self.right = right         # 右子树
        self.value = value         # 叶子节点的预测值
        self.samples = samples     # 节点包含的样本数

class HandwrittenDecisionTree:
    """手写决策树分类器"""
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.feature_names = None
        self.class_names = None
    
    def _gini_impurity(self, y):
        """计算基尼不纯度"""
        if len(y) == 0:
            return 0
        
        counts = np.bincount(y)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _entropy(self, y):
        """计算信息熵"""
        if len(y) == 0:
            return 0
        
        counts = np.bincount(y)
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]  # 避免log(0)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _impurity(self, y):
        """根据选择的标准计算不纯度"""
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
    
    def _information_gain(self, X, y, feature, threshold):
        """计算信息增益"""
        # 父节点不纯度
        parent_impurity = self._impurity(y)
        
        # 分割数据
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        # 子节点不纯度
        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        left_impurity = self._impurity(y[left_mask])
        right_impurity = self._impurity(y[right_mask])
        
        # 加权平均不纯度
        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        # 信息增益
        information_gain = parent_impurity - weighted_impurity
        return information_gain
    
    def _best_split(self, X, y):
        """找到最佳分割点"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            # 获取该特征的所有唯一值作为候选阈值
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            # 创建叶子节点
            most_common_class = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_class, samples=n_samples)
        
        # 找到最佳分割
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain == 0:
            # 无法进一步分割，创建叶子节点
            most_common_class = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_class, samples=n_samples)
        
        # 分割数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # 检查最小叶子节点样本数
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            most_common_class = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_class, samples=n_samples)
        
        # 递归构建左右子树
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_subtree, right=right_subtree, samples=n_samples)
    
    def fit(self, X, y, feature_names=None, class_names=None):
        """训练决策树"""
        self.feature_names = feature_names
        self.class_names = class_names
        self.root = self._build_tree(X, y)
    
    def _predict_sample(self, sample, node):
        """预测单个样本"""
        if node.value is not None:
            return node.value
        
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)
    
    def predict(self, X):
        """预测多个样本"""
        predictions = []
        for sample in X:
            prediction = self._predict_sample(sample, self.root)
            predictions.append(prediction)
        return np.array(predictions)
    
    def _get_tree_depth(self, node):
        """获取树的深度"""
        if node is None or node.value is not None:
            return 0
        return 1 + max(self._get_tree_depth(node.left), self._get_tree_depth(node.right))
    
    def _count_leaves(self, node):
        """计算叶子节点数量"""
        if node is None:
            return 0
        if node.value is not None:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
    
    def _count_nodes(self, node):
        """计算总节点数"""
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
    
    def get_tree_info(self):
        """获取树的信息"""
        depth = self._get_tree_depth(self.root)
        leaves = self._count_leaves(self.root)
        nodes = self._count_nodes(self.root)
        return depth, leaves, nodes
    
    def _print_tree(self, node, depth=0, prefix="Root: "):
        """打印决策树结构"""
        if node is not None:
            if node.value is not None:
                # 叶子节点
                class_name = self.class_names[node.value] if self.class_names else f"Class {node.value}"
                print("  " * depth + prefix + f"Predict {class_name} (samples: {node.samples})")
            else:
                # 内部节点
                feature_name = self.feature_names[node.feature] if self.feature_names else f"Feature {node.feature}"
                print("  " * depth + prefix + f"{feature_name} <= {node.threshold:.3f} (samples: {node.samples})")
                
                # 递归打印左右子树
                self._print_tree(node.left, depth + 1, "├─ True: ")
                self._print_tree(node.right, depth + 1, "└─ False: ")
    
    def print_tree(self):
        """打印完整的决策树"""
        print("决策树结构:")
        print("=" * 50)
        self._print_tree(self.root)
        print("=" * 50)

def calculate_accuracy(y_true, y_pred):
    """计算准确率"""
    return np.mean(y_true == y_pred)

def confusion_matrix_manual(y_true, y_pred, n_classes):
    """手动计算混淆矩阵"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm

def plot_confusion_matrix(cm, class_names):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('手写决策树混淆矩阵')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # 添加数值标签
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

# 主程序
if __name__ == "__main__":
    # 加载数据
    data = load_iris()
    X = data.data
    y = data.target
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("手写决策树分类器")
    print("=" * 50)
    
    # 创建并训练手写决策树
    dt = HandwrittenDecisionTree(
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion='gini'
    )
    
    dt.fit(X_train, y_train, feature_names=data.feature_names, class_names=data.target_names)
    
    # 预测
    predictions = dt.predict(X_test)
    
    # 计算准确率
    accuracy = calculate_accuracy(y_test, predictions)
    print(f"手写决策树准确率: {accuracy:.4f}")
    
    # 获取树的信息
    depth, leaves, nodes = dt.get_tree_info()
    print(f"树的深度: {depth}")
    print(f"叶子节点数: {leaves}")
    print(f"总节点数: {nodes}")
    
    # 打印决策树结构
    dt.print_tree()
    
    # 计算并显示混淆矩阵
    cm = confusion_matrix_manual(y_test, predictions, len(data.target_names))
    print("\n混淆矩阵:")
    print(cm)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, data.target_names)
    
    # 详细分类结果
    print("\n详细预测结果:")
    print("真实值 -> 预测值")
    for i, (true, pred) in enumerate(zip(y_test, predictions)):
        true_name = data.target_names[true]
        pred_name = data.target_names[pred]
        status = "✓" if true == pred else "✗"
        print(f"样本 {i+1:2d}: {true_name:10s} -> {pred_name:10s} {status}")
    
    # 测试不同参数的效果
    print("\n" + "=" * 50)
    print("测试不同参数的效果:")
    
    criterions = ['gini', 'entropy']
    max_depths = [3, 5, 10, None]
    
    for criterion in criterions:
        for max_depth in max_depths:
            dt_test = HandwrittenDecisionTree(
                max_depth=max_depth if max_depth else 20,
                criterion=criterion
            )
            dt_test.fit(X_train, y_train)
            pred_test = dt_test.predict(X_test)
            acc_test = calculate_accuracy(y_test, pred_test)
            depth_test, leaves_test, nodes_test = dt_test.get_tree_info()
            
            print(f"Criterion: {criterion:7s}, Max Depth: {str(max_depth):4s} -> "
                  f"Accuracy: {acc_test:.4f}, Depth: {depth_test:2d}, "
                  f"Leaves: {leaves_test:2d}, Nodes: {nodes_test:2d}") 