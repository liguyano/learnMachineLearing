import pygame
import sys
import math
from handwritten_decision_tree import HandwrittenDecisionTree, Node
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 初始化pygame
pygame.init()

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (100, 150, 255)
GREEN = (100, 255, 100)
RED = (255, 100, 100)
YELLOW = (255, 255, 100)
GRAY = (128, 128, 128)
LIGHT_BLUE = (200, 220, 255)
DARK_BLUE = (50, 100, 200)
PURPLE = (200, 100, 255)

# 屏幕设置
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Decision Tree Visualization")

# Font settings - Using Arial font
try:
    font_large = pygame.font.SysFont('arial', 24, bold=True)
    font_medium = pygame.font.SysFont('arial', 20)
    font_small = pygame.font.SysFont('arial', 16)
except:
    # Fallback to default font if Arial is not available
    font_large = pygame.font.Font(None, 24)
    font_medium = pygame.font.Font(None, 20)
    font_small = pygame.font.Font(None, 16)

class TreeVisualizer:
    def __init__(self, decision_tree, feature_names, class_names):
        self.tree = decision_tree
        self.feature_names = feature_names
        self.class_names = class_names
        self.node_positions = {}
        self.node_radius = 40
        self.level_height = 140
        self.min_horizontal_spacing = 120
        self.selected_node = None
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 50
        self.dragging = False
        self.drag_start_pos = (0, 0)
        self.drag_start_offset = (0, 0)
        
    def calculate_tree_width(self, node, depth=0):
        """计算树的宽度"""
        if node is None or node.value is not None:
            return 1
        
        left_width = self.calculate_tree_width(node.left, depth + 1)
        right_width = self.calculate_tree_width(node.right, depth + 1)
        return left_width + right_width
    
    def calculate_positions(self, node, x, y, width, depth=0):
        """Recursively calculate node positions with improved spacing"""
        if node is None:
            return
        
        # Store current node position
        self.node_positions[id(node)] = (x, y, depth)
        
        if node.value is not None:  # Leaf node
            return
        
        # Calculate left and right subtree widths
        left_width = self.calculate_tree_width(node.left)
        right_width = self.calculate_tree_width(node.right)
        total_width = left_width + right_width
        
        if total_width > 0:
            # Improved spacing calculation to prevent collisions
            min_separation = self.node_radius * 2.5  # Minimum distance between nodes
            
            # Calculate base positions
            left_ratio = left_width / total_width
            right_ratio = right_width / total_width
            
            # Ensure minimum separation between child nodes
            base_separation = max(width * 0.6, min_separation * total_width)
            
            left_x = x - (base_separation * left_ratio) / 2
            right_x = x + (base_separation * right_ratio) / 2
            
            # Ensure nodes don't get too close
            if abs(right_x - left_x) < min_separation:
                adjustment = (min_separation - abs(right_x - left_x)) / 2
                left_x -= adjustment
                right_x += adjustment
            
            child_y = y + self.level_height
            
            # Calculate child width based on depth to prevent overcrowding
            child_width = max(width * 0.75, min_separation * max(left_width, right_width))
            
            # Recursively calculate child positions
            self.calculate_positions(node.left, left_x, child_y, child_width, depth + 1)
            self.calculate_positions(node.right, right_x, child_y, child_width, depth + 1)
    
    def draw_node(self, screen, node, x, y, depth):
        """绘制单个节点"""
        # 应用缩放和偏移
        draw_x = int((x + self.offset_x) * self.zoom)
        draw_y = int((y + self.offset_y) * self.zoom)
        radius = int(self.node_radius * self.zoom)
        
        # 检查节点是否在屏幕范围内
        if (draw_x < -radius or draw_x > SCREEN_WIDTH + radius or 
            draw_y < -radius or draw_y > SCREEN_HEIGHT + radius):
            return
        
        # 选择节点颜色
        if node.value is not None:  # 叶子节点
            if node.value == 0:
                color = RED
            elif node.value == 1:
                color = GREEN
            else:
                color = BLUE
        else:  # 内部节点
            color = LIGHT_BLUE
        
        # 如果是选中的节点，使用特殊颜色
        if self.selected_node == id(node):
            color = YELLOW
        
        # 绘制节点圆圈
        pygame.draw.circle(screen, color, (draw_x, draw_y), radius)
        pygame.draw.circle(screen, BLACK, (draw_x, draw_y), radius, 2)
        
        # 绘制节点文本
        if node.value is not None:  # 叶子节点
            class_name = self.class_names[node.value]
            text = font_small.render(class_name[:8], True, BLACK)
            text_rect = text.get_rect(center=(draw_x, draw_y - 5))
            screen.blit(text, text_rect)
            
            # 显示样本数
            samples_text = font_small.render(f"n={node.samples}", True, BLACK)
            samples_rect = samples_text.get_rect(center=(draw_x, draw_y + 10))
            screen.blit(samples_text, samples_rect)
        else:  # 内部节点
            feature_name = self.feature_names[node.feature]
            # 缩短特征名
            if len(feature_name) > 10:
                feature_name = feature_name[:8] + ".."
            
            text = font_small.render(feature_name, True, BLACK)
            text_rect = text.get_rect(center=(draw_x, draw_y - 8))
            screen.blit(text, text_rect)
            
            threshold_text = font_small.render(f"≤{node.threshold:.2f}", True, BLACK)
            threshold_rect = threshold_text.get_rect(center=(draw_x, draw_y + 8))
            screen.blit(threshold_text, threshold_rect)
    
    def draw_edge(self, screen, parent_pos, child_pos, is_left_child):
        """绘制边"""
        px, py = parent_pos
        cx, cy = child_pos
        
        # 应用缩放和偏移
        px = int((px + self.offset_x) * self.zoom)
        py = int((py + self.offset_y) * self.zoom)
        cx = int((cx + self.offset_x) * self.zoom)
        cy = int((cy + self.offset_y) * self.zoom)
        
        # 检查边是否在屏幕范围内
        if (max(px, cx) < 0 or min(px, cx) > SCREEN_WIDTH or
            max(py, cy) < 0 or min(py, cy) > SCREEN_HEIGHT):
            return
        
        # 绘制连线
        color = GREEN if is_left_child else RED
        pygame.draw.line(screen, color, (px, py), (cx, cy), 2)
        
        # 在边上添加标签
        mid_x = (px + cx) // 2
        mid_y = (py + cy) // 2
        
        label = "True" if is_left_child else "False"
        text = font_small.render(label, True, color)
        text_rect = text.get_rect(center=(mid_x, mid_y))
        
        # 添加白色背景
        pygame.draw.rect(screen, WHITE, text_rect.inflate(4, 2))
        screen.blit(text, text_rect)
    
    def draw_tree_recursive(self, screen, node):
        """递归绘制整个树"""
        if node is None or id(node) not in self.node_positions:
            return
        
        x, y, depth = self.node_positions[id(node)]
        
        # 先绘制到子节点的边
        if node.value is None:  # 不是叶子节点
            if node.left and id(node.left) in self.node_positions:
                left_x, left_y, _ = self.node_positions[id(node.left)]
                self.draw_edge(screen, (x, y), (left_x, left_y), True)
                self.draw_tree_recursive(screen, node.left)
            
            if node.right and id(node.right) in self.node_positions:
                right_x, right_y, _ = self.node_positions[id(node.right)]
                self.draw_edge(screen, (x, y), (right_x, right_y), False)
                self.draw_tree_recursive(screen, node.right)
        
        # 最后绘制当前节点（这样节点会显示在边的上方）
        self.draw_node(screen, node, x, y, depth)
    
    def get_node_at_position(self, pos, node):
        """获取指定位置的节点"""
        if node is None or id(node) not in self.node_positions:
            return None
        
        x, y, depth = self.node_positions[id(node)]
        draw_x = int((x + self.offset_x) * self.zoom)
        draw_y = int((y + self.offset_y) * self.zoom)
        radius = int(self.node_radius * self.zoom)
        
        # 检查点击位置是否在节点内
        distance = math.sqrt((pos[0] - draw_x)**2 + (pos[1] - draw_y)**2)
        if distance <= radius:
            return node
        
        # 递归检查子节点
        if node.left:
            result = self.get_node_at_position(pos, node.left)
            if result:
                return result
        
        if node.right:
            result = self.get_node_at_position(pos, node.right)
            if result:
                return result
        
        return None
    
    def draw_info_panel(self, screen):
        """Draw information panel"""
        panel_rect = pygame.Rect(10, 10, 300, 150)
        pygame.draw.rect(screen, WHITE, panel_rect)
        pygame.draw.rect(screen, BLACK, panel_rect, 2)
        
        # Title
        title = font_large.render("Tree Information", True, BLACK)
        screen.blit(title, (20, 20))
        
        # Tree statistics
        depth, leaves, nodes = self.tree.get_tree_info()
        info_texts = [
            f"Tree Depth: {depth}",
            f"Leaf Nodes: {leaves}",
            f"Total Nodes: {nodes}",
            f"Zoom: {self.zoom:.1f}x"
        ]
        
        for i, text in enumerate(info_texts):
            rendered = font_medium.render(text, True, BLACK)
            screen.blit(rendered, (20, 50 + i * 25))
        
        # Selected node information
        if self.selected_node:
            selected_rect = pygame.Rect(10, 170, 300, 100)
            pygame.draw.rect(screen, LIGHT_BLUE, selected_rect)
            pygame.draw.rect(screen, BLACK, selected_rect, 2)
            
            title = font_medium.render("Selected Node Info", True, BLACK)
            screen.blit(title, (20, 180))
            
            # Find the selected node
            selected_node_obj = self.find_node_by_id(self.tree.root, self.selected_node)
            if selected_node_obj:
                if selected_node_obj.value is not None:
                    info = f"Class: {self.class_names[selected_node_obj.value]}"
                else:
                    feature_name = self.feature_names[selected_node_obj.feature]
                    info = f"Feature: {feature_name}"
                    info2 = f"Threshold: {selected_node_obj.threshold:.3f}"
                    rendered2 = font_small.render(info2, True, BLACK)
                    screen.blit(rendered2, (20, 220))
                
                rendered = font_small.render(info, True, BLACK)
                screen.blit(rendered, (20, 200))
                
                samples_info = f"Samples: {selected_node_obj.samples}"
                rendered3 = font_small.render(samples_info, True, BLACK)
                screen.blit(rendered3, (20, 240))
    
    def find_node_by_id(self, node, node_id):
        """根据ID查找节点"""
        if node is None:
            return None
        if id(node) == node_id:
            return node
        
        left_result = self.find_node_by_id(node.left, node_id)
        if left_result:
            return left_result
        
        return self.find_node_by_id(node.right, node_id)
    
    def detect_and_fix_collisions(self):
        """Detect and fix node collisions by adjusting positions"""
        positions_by_level = {}
        
        # Group nodes by level
        for node_id, (x, y, depth) in self.node_positions.items():
            if depth not in positions_by_level:
                positions_by_level[depth] = []
            positions_by_level[depth].append((node_id, x, y))
        
        # Check and fix collisions at each level
        for depth, nodes in positions_by_level.items():
            if len(nodes) <= 1:
                continue
                
            # Sort nodes by x position
            nodes.sort(key=lambda item: item[1])
            
            # Adjust positions to prevent overlaps
            min_distance = self.node_radius * 2.2
            
            for i in range(1, len(nodes)):
                prev_node = nodes[i-1]
                curr_node = nodes[i]
                
                prev_x = prev_node[1]
                curr_x = curr_node[1]
                
                if curr_x - prev_x < min_distance:
                    # Adjust current node position
                    new_x = prev_x + min_distance
                    nodes[i] = (curr_node[0], new_x, curr_node[2])
                    
                    # Update the position in the main dictionary
                    self.node_positions[curr_node[0]] = (new_x, curr_node[2], depth)
    
    def draw_controls(self, screen):
        """Draw control instructions"""
        controls_rect = pygame.Rect(SCREEN_WIDTH - 250, 10, 240, 140)
        pygame.draw.rect(screen, WHITE, controls_rect)
        pygame.draw.rect(screen, BLACK, controls_rect, 2)
        
        title = font_medium.render("Controls", True, BLACK)
        screen.blit(title, (SCREEN_WIDTH - 240, 20))
        
        controls = [
            "Left Click: Select node",
            "Right Click + Drag: Pan view",
            "Mouse Wheel: Zoom",
            "WASD: Move view",
            "R: Reset view",
            "ESC: Exit"
        ]
        
        for i, control in enumerate(controls):
            rendered = font_small.render(control, True, BLACK)
            screen.blit(rendered, (SCREEN_WIDTH - 240, 45 + i * 18))
    
    def draw(self, screen):
        """Draw the entire visualization"""
        screen.fill(WHITE)
        
        # Calculate node positions (if not calculated yet)
        if not self.node_positions:
            tree_width = self.calculate_tree_width(self.tree.root) * self.min_horizontal_spacing
            # Ensure minimum width to prevent collisions
            tree_width = max(tree_width, SCREEN_WIDTH * 0.8)
            start_x = SCREEN_WIDTH // 2
            start_y = 100
            self.calculate_positions(self.tree.root, start_x, start_y, tree_width)
            
            # Apply collision detection and fix overlaps
            self.detect_and_fix_collisions()
        
        # Draw tree
        self.draw_tree_recursive(screen, self.tree.root)
        
        # Draw information panel
        self.draw_info_panel(screen)
        
        # Draw control instructions
        self.draw_controls(screen)

def main():
    # Load data and train decision tree
    data = load_iris()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create handwritten decision tree
    dt = HandwrittenDecisionTree(max_depth=5, min_samples_split=2, min_samples_leaf=1)
    dt.fit(X_train, y_train, feature_names=data.feature_names, class_names=data.target_names)
    
    # Create visualizer
    visualizer = TreeVisualizer(dt, data.feature_names, data.target_names)
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset view
                    visualizer.zoom = 1.0
                    visualizer.offset_x = 0
                    visualizer.offset_y = 50
                    visualizer.dragging = False
                elif event.key == pygame.K_w:
                    visualizer.offset_y += 20
                elif event.key == pygame.K_s:
                    visualizer.offset_y -= 20
                elif event.key == pygame.K_a:
                    visualizer.offset_x += 20
                elif event.key == pygame.K_d:
                    visualizer.offset_x -= 20
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    clicked_node = visualizer.get_node_at_position(event.pos, dt.root)
                    if clicked_node:
                        visualizer.selected_node = id(clicked_node)
                    else:
                        visualizer.selected_node = None
                
                elif event.button == 3:  # Right click - start dragging
                    visualizer.dragging = True
                    visualizer.drag_start_pos = event.pos
                    visualizer.drag_start_offset = (visualizer.offset_x, visualizer.offset_y)
                
                elif event.button == 4:  # Scroll up
                    visualizer.zoom = min(2.0, visualizer.zoom + 0.1)
                elif event.button == 5:  # Scroll down
                    visualizer.zoom = max(0.3, visualizer.zoom - 0.1)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:  # Right click release - stop dragging
                    visualizer.dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if visualizer.dragging:
                    # Calculate drag offset
                    current_pos = event.pos
                    dx = current_pos[0] - visualizer.drag_start_pos[0]
                    dy = current_pos[1] - visualizer.drag_start_pos[1]
                    
                    # Update offset based on drag distance
                    visualizer.offset_x = visualizer.drag_start_offset[0] + dx / visualizer.zoom
                    visualizer.offset_y = visualizer.drag_start_offset[1] + dy / visualizer.zoom
        
        # Draw
        visualizer.draw(screen)
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 