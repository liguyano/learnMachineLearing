import pygame
import sys
import numpy as np

# 初始化 pygame
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("多层决策树可视化 (Pygame)")
font = pygame.font.SysFont("SimHei", 14)  # 支持中文
clock = pygame.time.Clock()

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (100, 150, 255)

# 计算基尼不纯度
def gini_impurity(groups, y):
    n = len(y)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        label_counts = {}
        for idx in group:
            label = y[idx]
            label_counts[label] = label_counts.get(label, 0) + 1
        impurity = 1.0
        for label in label_counts:
            prob = label_counts[label] / size
            impurity -= prob ** 2
        gini += impurity * (size / n)
    return gini

# 找到最佳分裂点
def get_split(X, y):
    best_score = float('inf')
    best_feature = None
    best_threshold = None
    best_groups = None

    n_samples, n_features = X.shape

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left = []
            right = []
            for i in range(n_samples):
                if X[i, feature] <= threshold:
                    left.append(i)
                else:
                    right.append(i)
            score = gini_impurity([left, right], y)
            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold
                best_groups = [left, right]

    return {
        'feature': best_feature,
        'threshold': best_threshold,
        'groups': best_groups,
        'gini': best_score
    }

# 构建树（带坐标）
def build_tree(X, y, max_depth, depth, node_id=0, x=WIDTH//2, y_pos=50, level_width=WIDTH//2):
    split = get_split(X, y)
    left, right = split['groups']

    left_y = y[left]
    right_y = y[right]

    node = {
        'id': node_id,
        'feature': split['feature'],
        'threshold': split['threshold'],
        'gini': split['gini'],
        'n_samples': len(y),
        'value': dict(zip(*np.unique(y, return_counts=True))),
        'left': None,
        'right': None,
        'x': x,
        'y': y_pos,
        'children_pos': []
    }

    print(f"Node {node_id}: Split on feature {split['feature']} at threshold {split['threshold']:.3f}, Gini: {split['gini']:.3f}")

    # 停止条件
    if depth >= max_depth or len(left) == 0 or len(right) == 0 or len(np.unique(left_y)) == 1 or len(np.unique(right_y)) == 1:
        node['left'] = {
            'id': node_id * 2 + 1,
            'value': int(np.round(np.mean(left_y))) if len(left_y) > 0 else 0,
            'leaf': True,
            'n_samples': len(left_y),
            'class': int(np.round(np.mean(left_y))) if len(left_y) > 0 else 0,
            'x': x - level_width / 2,
            'y': y_pos + 100
        }
        node['right'] = {
            'id': node_id * 2 + 2,
            'value': int(np.round(np.mean(right_y))) if len(right_y) > 0 else 0,
            'leaf': True,
            'n_samples': len(right_y),
            'class': int(np.round(np.mean(right_y))) if len(right_y) > 0 else 0,
            'x': x + level_width / 2,
            'y': y_pos + 100
        }
        node['children_pos'] = [(node['left']['x'], node['left']['y']), (node['right']['x'], node['right']['y'])]
        return node

    node['left'] = build_tree(X[left], y[left], max_depth, depth + 1, node_id * 2 + 1, x - level_width / 2, y_pos + 100, level_width / 2)
    node['right'] = build_tree(X[right], y[right], max_depth, depth + 1, node_id * 2 + 2, x + level_width / 2, y_pos + 100, level_width / 2)
    node['children_pos'] = [(node['left']['x'], node['left']['y']), (node['right']['x'], node['right']['y'])]

    return node

# 绘制节点
def draw_node(node):
    is_leaf = 'leaf' in node and node['leaf']
    color = BLUE if is_leaf else GRAY
    pygame.draw.circle(screen, color, (node['x'], node['y']), 30)
    pygame.draw.circle(screen, BLACK, (node['x'], node['y']), 30, 2)

    if is_leaf:
        text = font.render(f"Class: {node['class']}", True, BLACK)
    else:
        text = font.render(f"X[{node['feature']}] ≤ {node['threshold']:.2f}", True, BLACK)
    screen.blit(text, (node['x'] - text.get_width() // 2, node['y'] - text.get_height() // 2 + 5))

# 绘制树结构
def draw_tree(node):
    draw_node(node)
    if 'left' in node and node['left']:
        pygame.draw.line(screen, BLACK, (node['x'], node['y']), (node['left']['x'], node['left']['y']), 2)
        draw_tree(node['left'])
    if 'right' in node and node['right']:
        pygame.draw.line(screen, BLACK, (node['x'], node['y']), (node['right']['x'], node['right']['y']), 2)
        draw_tree(node['right'])

# 示例数据集
dataset = np.array([
    [2.771, 1.789, 0],
    [1.728, 1.169, 0],
    [3.678, 2.812, 0],
    [5.596, 2.008, 1],
    [6.679, 3.656, 1],
    [7.792, 2.495, 1]
])

X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

# 构建树（max_depth=5）
tree = build_tree(X, y, max_depth=5, depth=1)

# 主循环
running = True
while running:
    screen.fill(WHITE)
    draw_tree(tree)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()