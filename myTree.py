import numpy as np
import pandas as pd
import random

# 设置随机种子（可复现）
np.random.seed(42)
random.seed(42)

# 生成 100 个样本
n_samples = 100

# 生成年龄：18-65 岁
age = np.random.randint(18, 66, size=n_samples)

# 生成身高：150-190 cm
height = np.random.randint(150, 191, size=n_samples)

# 生成体重：基于身高大致合理，加入随机性
# BMI 在 18-30 之间随机，然后计算体重
bmi = np.random.uniform(18, 30, size=n_samples)
weight = bmi * (height / 100) ** 2
weight = np.round(weight, 1)  # 保留一位小数

# 设计“是否购买”的逻辑（模拟真实行为）
purchased = []
for i in range(n_samples):
    a = age[i]
    w = weight[i]
    h = height[i]

    # 年轻人 + 体重偏高 → 更可能购买
    score = 0
    if a < 30:
        score += 1.0
    elif a < 45:
        score += 0.5
    else:
        score += 0.1

    # BMI 偏高 → 更可能购买
    bmi_i = w / (h / 100) ** 2
    if bmi_i > 25:
        score += 1.2
    elif bmi_i > 22:
        score += 0.6

    # 身高较高也可能影响
    if h > 175:
        score += 0.3

    # 转换为概率
    prob = 1 / (1 + np.exp(- (score - 2.0)))  # Sigmoid
    p = 1 if random.random() < prob else 0
    purchased.append(p)

# 构建成 DataFrame
data = pd.DataFrame({
    'Age': age,
    'Weight': weight,
    'Height': height,
    'Purchased': purchased
})

# 查看前 10 行
print(data.head(10))
print(data.at[50,'Purchased'])
t_age=18
t_height=170
t_weight=50
last_gini=1
t_mea=1
m_age=15
last_mean=1
def gini_impurity(sample):
    """
    计算样本集中 'Purchased' 类别的基尼不纯度
    参数:
        sample: pandas DataFrame 或 Series，包含 'Purchased' 列
    返回:
        gini: 基尼不纯度，浮点数（0 ~ 0.5 之间）
    """
    # 如果输入是 DataFrame，提取 'Purchased' 列
    if hasattr(sample, 'Purchased'):
        labels = sample['Purchased']
    else:
        labels = sample  # 假设输入已经是标签数组

    # 统计每个类别的数量
    counts = np.bincount(labels)  # 只适用于类别为 0,1,2... 的整数

    # 总样本数
    n_total = len(labels)

    # 如果没有样本，返回 0（纯）
    if n_total == 0:
        return 0.0

    # 计算每个类别的比例
    probabilities = counts / n_total

    # 计算基尼不纯度
    gini = 1 - np.sum(probabilities ** 2)

    return gini


min_left=1
for i in range(100):
    sample=data.sample(40)
    left=sample[sample['Age']<=t_age]
    right=sample[sample['Age']>t_age]
    left_geni=gini_impurity(left)
    right_geni=gini_impurity(right)
    if len(left) == 0 or len(right) == 0:
        continue  # 跳过无效分裂
    if left_geni<= min_left:
        min_left=left_geni
        m_age=t_age
        print(min_left, m_age)
    t_age+=0.5

print(min_left,m_age)

