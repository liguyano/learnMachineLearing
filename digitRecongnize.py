# 导入必要的库
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import matplotlib

# 1. 加载数据集
digits = datasets.load_digits()
# 设置中文字体和解决负号显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题
# 打印数据集信息
print("特征数据形状:", digits.data.shape)   # (n_samples, n_features)
print("标签数据形状:", digits.target.shape) # (n_samples, )

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42
)

# 3. 特征标准化（Standardization）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 创建并训练 SVM 分类模型
model = SVC(gamma='scale')  # 使用默认参数
model.fit(X_train, y_train)

# 5. 在测试集上进行预测
y_pred = model.predict(X_test)

# 6. 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy * 100:.2f}%")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 7. 可视化部分预测结果
def plot_gallery(images, titles, h, w, n_row=3, n_col=5):
    """显示图像画廊"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.axis('off')

# 显示前 15 张图片及其预测结果
prediction_titles = [f'Pred: {pred}' for pred in y_pred[:15]]
plot_gallery(X_test, prediction_titles, 8, 8)

plt.suptitle("手写数字识别预测结果", fontsize=16)
plt.show()