import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from handwritten_decision_tree import HandwrittenDecisionTree, Node
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math

class PlotlyTreeVisualizer:
    def __init__(self, decision_tree, feature_names, class_names):
        self.tree = decision_tree
        self.feature_names = feature_names
        self.class_names = class_names
        self.node_positions = {}
        self.edges = []
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
        
    def calculate_tree_layout(self, node, x=0, y=0, layer_width=4, depth=0):
        """Calculate positions for tree nodes using a layered approach"""
        if node is None:
            return
        
        # Store position
        self.node_positions[id(node)] = {'x': x, 'y': y, 'depth': depth}
        
        if node.value is not None:  # Leaf node
            return
        
        # Calculate positions for children
        next_layer_width = layer_width / 2
        child_y = y - 1
        
        if node.left:
            left_x = x - layer_width / 2
            self.edges.append({
                'x0': x, 'y0': y, 'x1': left_x, 'y1': child_y,
                'parent_id': id(node), 'child_id': id(node.left),
                'label': 'True'
            })
            self.calculate_tree_layout(node.left, left_x, child_y, next_layer_width, depth + 1)
        
        if node.right:
            right_x = x + layer_width / 2
            self.edges.append({
                'x0': x, 'y0': y, 'x1': right_x, 'y1': child_y,
                'parent_id': id(node), 'child_id': id(node.right),
                'label': 'False'
            })
            self.calculate_tree_layout(node.right, right_x, child_y, next_layer_width, depth + 1)
    
    def get_node_info(self, node):
        """Get formatted information about a node"""
        if node.value is not None:
            return {
                'text': f"Class: {self.class_names[node.value]}<br>Samples: {node.samples}",
                'color': self.colors[node.value],
                'type': 'leaf'
            }
        else:
            feature_name = self.feature_names[node.feature]
            return {
                'text': f"{feature_name}<br>≤ {node.threshold:.3f}<br>Samples: {node.samples}",
                'color': '#E8E8E8',
                'type': 'internal'
            }
    
    def create_interactive_tree(self):
        """Create an interactive plotly tree visualization"""
        # Calculate layout
        self.calculate_tree_layout(self.tree.root, x=0, y=0, layer_width=8)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        hover_text = []
        
        def collect_node_data(node):
            if node is None:
                return
            
            pos = self.node_positions[id(node)]
            info = self.get_node_info(node)
            
            node_x.append(pos['x'])
            node_y.append(pos['y'])
            node_text.append(info['text'])
            node_colors.append(info['color'])
            
            # Size based on number of samples
            size = max(20, min(60, node.samples * 2))
            node_sizes.append(size)
            
            # Detailed hover information
            if node.value is not None:
                hover_info = f"Leaf Node<br>Class: {self.class_names[node.value]}<br>Samples: {node.samples}<br>Depth: {pos['depth']}"
            else:
                hover_info = f"Decision Node<br>Feature: {self.feature_names[node.feature]}<br>Threshold: {node.threshold:.3f}<br>Samples: {node.samples}<br>Depth: {pos['depth']}"
            hover_text.append(hover_info)
            
            collect_node_data(node.left)
            collect_node_data(node.right)
        
        collect_node_data(self.tree.root)
        
        # Create the figure
        fig = go.Figure()
        
        # Add edges
        for edge in self.edges:
            fig.add_trace(go.Scatter(
                x=[edge['x0'], edge['x1']],
                y=[edge['y0'], edge['y1']],
                mode='lines',
                line=dict(color='#666666', width=2),
                hoverinfo='none',
                showlegend=False
            ))
            
            # Add edge labels
            mid_x = (edge['x0'] + edge['x1']) / 2
            mid_y = (edge['y0'] + edge['y1']) / 2
            fig.add_trace(go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode='text',
                text=[edge['label']],
                textfont=dict(size=10, color='#333333'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='#333333'),
                opacity=0.8
            ),
            text=node_text,
            textposition='middle center',
            textfont=dict(size=10, color='black'),
            hovertext=hover_text,
            hoverinfo='text',
            name='Decision Tree Nodes'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Interactive Decision Tree Visualization',
                'x': 0.5,
                'font': {'size': 20}
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Hover over nodes for details • Click and drag to pan • Scroll to zoom",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002, xanchor='left', yanchor='bottom',
                    font=dict(size=12, color='#666666')
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=700
        )
        
        return fig
    
    def create_feature_importance_chart(self):
        """Create a feature importance visualization"""
        # Calculate feature importance from the tree
        feature_counts = {i: 0 for i in range(len(self.feature_names))}
        
        def count_features(node):
            if node is None or node.value is not None:
                return
            feature_counts[node.feature] += node.samples
            count_features(node.left)
            count_features(node.right)
        
        count_features(self.tree.root)
        
        # Normalize to get importance scores
        total_samples = sum(feature_counts.values())
        if total_samples > 0:
            importance_scores = [count / total_samples for count in feature_counts.values()]
        else:
            importance_scores = [0] * len(self.feature_names)
        
        fig = go.Figure(data=[
            go.Bar(
                x=self.feature_names,
                y=importance_scores,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                text=[f'{score:.3f}' for score in importance_scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Feature Importance in Decision Tree',
            xaxis_title='Features',
            yaxis_title='Importance Score',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_tree_statistics_dashboard(self):
        """Create a dashboard with tree statistics"""
        depth, leaves, nodes = self.tree.get_tree_info()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Tree Statistics', 'Node Distribution by Depth', 
                          'Class Distribution in Leaves', 'Sample Distribution'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "histogram"}]]
        )
        
        # Tree statistics indicators
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = depth,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Tree Depth<br>Leaves: {leaves}<br>Total Nodes: {nodes}"},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': "#4ECDC4"},
                'steps': [
                    {'range': [0, 3], 'color': "#E8F5E8"},
                    {'range': [3, 6], 'color': "#C8E6C9"},
                    {'range': [6, 10], 'color': "#A5D6A7"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 8}}
        ), row=1, col=1)
        
        # Node distribution by depth
        depth_counts = {}
        def count_by_depth(node, depth=0):
            if node is None:
                return
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            count_by_depth(node.left, depth + 1)
            count_by_depth(node.right, depth + 1)
        
        count_by_depth(self.tree.root)
        
        fig.add_trace(go.Bar(
            x=list(depth_counts.keys()),
            y=list(depth_counts.values()),
            marker_color='#45B7D1',
            name='Nodes per Depth'
        ), row=1, col=2)
        
        # Class distribution in leaves
        leaf_classes = []
        def collect_leaf_classes(node):
            if node is None:
                return
            if node.value is not None:
                leaf_classes.append(self.class_names[node.value])
            else:
                collect_leaf_classes(node.left)
                collect_leaf_classes(node.right)
        
        collect_leaf_classes(self.tree.root)
        class_counts = pd.Series(leaf_classes).value_counts()
        
        fig.add_trace(go.Pie(
            labels=class_counts.index,
            values=class_counts.values,
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1']
        ), row=2, col=1)
        
        # Sample distribution
        sample_sizes = []
        def collect_sample_sizes(node):
            if node is None:
                return
            sample_sizes.append(node.samples)
            collect_sample_sizes(node.left)
            collect_sample_sizes(node.right)
        
        collect_sample_sizes(self.tree.root)
        
        fig.add_trace(go.Histogram(
            x=sample_sizes,
            nbinsx=10,
            marker_color='#96CEB4',
            name='Sample Distribution'
        ), row=2, col=2)
        
        fig.update_layout(
            title_text="Decision Tree Analysis Dashboard",
            showlegend=False,
            height=600
        )
        
        return fig

def main():
    print("Creating Plotly Decision Tree Visualizations...")
    
    # Load data and train decision tree
    data = load_iris()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create handwritten decision tree
    dt = HandwrittenDecisionTree(max_depth=4, min_samples_split=3, min_samples_leaf=2)
    dt.fit(X_train, y_train, feature_names=data.feature_names, class_names=data.target_names)
    
    # Create visualizer
    visualizer = PlotlyTreeVisualizer(dt, data.feature_names, data.target_names)
    
    # Create visualizations
    print("1. Creating interactive tree visualization...")
    tree_fig = visualizer.create_interactive_tree()
    tree_fig.show()
    
    print("2. Creating feature importance chart...")
    importance_fig = visualizer.create_feature_importance_chart()
    importance_fig.show()
    
    print("3. Creating tree statistics dashboard...")
    dashboard_fig = visualizer.create_tree_statistics_dashboard()
    dashboard_fig.show()
    
    # Save visualizations as HTML files
    print("\nSaving visualizations as HTML files...")
    tree_fig.write_html("decision_tree_interactive.html")
    importance_fig.write_html("feature_importance.html")
    dashboard_fig.write_html("tree_dashboard.html")
    
    print("✅ All visualizations created and saved!")
    print("📁 Files saved:")
    print("   - decision_tree_interactive.html")
    print("   - feature_importance.html") 
    print("   - tree_dashboard.html")
    
    # Print tree performance
    predictions = dt.predict(X_test)
    accuracy = np.mean(y_test == predictions)
    print(f"\n🎯 Tree Accuracy: {accuracy:.4f}")
    
    depth, leaves, nodes = dt.get_tree_info()
    print(f"📊 Tree Stats: Depth={depth}, Leaves={leaves}, Nodes={nodes}")

if __name__ == "__main__":
    main() 