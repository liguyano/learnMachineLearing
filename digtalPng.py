from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 加载数据集
digits = datasets.load_digits()

# 数据和标签
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 创建分类器
classifier = svm.SVC(gamma=0.001)

# 训练模型
classifier.fit(X_train_scaled, y_train)

# 保存模型和scaler
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(classifier, 'digit_model.pkl')
from PIL import Image
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 加载模型和scaler
model = joblib.load('digit_model.pkl')
scaler = joblib.load('scaler.pkl')

# 加载并预处理你的PNG图像（请替换成你自己的图片路径）
image_path = 'your_digit_image.png'  # 替换为你的PNG文件路径
image = Image.open(image_path).convert('L')  # 转换为灰度图

# 调整大小为8x8（根据load_digits的数据调整）
image = image.resize((8, 8), Image.Resampling.LANCZOS)

# 将像素值从0~255映射到0~16
image_data = np.array(image)
image_data = np.invert(image_data)  # 反色（白底黑字→黑底白字）
image_data = (image_data / 255.0) * 16
image_data = image_data.astype(np.uint8)

# 展平成一维数组(64,)
image_flattened = image_data.reshape(1, -1)

# 使用训练时的scaler进行特征缩放
image_scaled = scaler.transform(image_flattened)

# 预测
predicted_digit = model.predict(image_scaled)
print(f"预测结果: {predicted_digit[0]}")

# 可视化
plt.imshow(image_data, cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit[0]}")
plt.axis('off')
plt.show()