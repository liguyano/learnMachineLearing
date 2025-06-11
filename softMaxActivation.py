import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # 提高数值稳定性
    return e_x / e_x.sum(axis=0)

# 示例输入
scores = np.array([3.0, 1.0, 0.2])
print(softmax(scores))