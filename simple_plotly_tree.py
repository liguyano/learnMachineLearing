import plotly.graph_objects as go
import plotly.offline as pyo
from handwritten_decision_tree import HandwrittenDecisionTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

def create_simple_tree_plot():
    """Create a simple, clear decision tree visualization"""
    
    # Load data and create tree
    print("Loading data and training tree...")
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # Create a smaller tree for better visibility
    dt = HandwrittenDecisionTree(max_depth=3, min_samples_split=5, min_samples_leaf=2)
    dt.fit(X_train, y_train, feature_names=data.feature_names, class_names=data.target_names)
    
    print("Tree created! Building visualization...")
    
    # Collect all nodes and their information
    nodes = []
    edges = []
    
    def traverse_tree(node, x=0, y=0, level=0, parent_x=None, parent_y=None):
        if node is None:
            return
        
        # Store node information
        if node.value is not None:  # Leaf node
            node_info = {
                'x': x, 'y': y, 'level': level,
                'text': f"Class: {data.target_names[node.value]}",
                'samples': node.samples,
                'color': ['red', 'green', 'blue'][node.value],
                'is_leaf': True
            }
        else:  # Internal node
            feature_name = data.feature_names[node.feature]
            node_info = {
                'x': x, 'y': y, 'level': level,
                'text': f"{feature_name}<br>≤ {node.threshold:.2f}",
                'samples': node.samples,
                'color': 'lightgray',
                'is_leaf': False
            }
        
        nodes.append(node_info)
        
        # Add edge from parent
        if parent_x is not None and parent_y is not None:
            edges.append({
                'x0': parent_x, 'y0': parent_y,
                'x1': x, 'y1': y
            })
        
        # Recursively process children
        if not node_info['is_leaf']:
            # Left child (True branch)
            traverse_tree(node.left, x - 2**(2-level), y - 1, level + 1, x, y)
            # Right child (False branch)  
            traverse_tree(node.right, x + 2**(2-level), y - 1, level + 1, x, y)
    
    # Build the tree structure
    traverse_tree(dt.root, x=0, y=0)
    
    print(f"Found {len(nodes)} nodes and {len(edges)} edges")
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges first (so they appear behind nodes)
    for edge in edges:
        fig.add_trace(go.Scatter(
            x=[edge['x0'], edge['x1']],
            y=[edge['y0'], edge['y1']],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add nodes
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node['x']],
            y=[node['y']],
            mode='markers+text',
            marker=dict(
                size=max(30, node['samples']),
                color=node['color'],
                line=dict(width=2, color='black')
            ),
            text=node['text'],
            textposition='middle center',
            textfont=dict(size=10, color='black'),
            showlegend=False,
            hovertemplate=f"<b>{node['text']}</b><br>Samples: {node['samples']}<br>Level: {node['level']}<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Decision Tree Structure',
            'x': 0.5,
            'font': {'size': 24}
        },
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-8, 8]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-4, 1]
        ),
        plot_bgcolor='white',
        width=1000,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    print("Displaying tree visualization...")
    
    # Show the plot
    fig.show()
    
    # Also save as HTML
    fig.write_html("simple_decision_tree.html")
    print("Tree saved as 'simple_decision_tree.html'")
    
    # Print tree information
    print(f"\n🌳 Tree Information:")
    print(f"   Accuracy: {np.mean(dt.predict(X_test) == y_test):.4f}")
    depth, leaves, nodes_count = dt.get_tree_info()
    print(f"   Depth: {depth}, Leaves: {leaves}, Total Nodes: {nodes_count}")
    
    return fig

if __name__ == "__main__":
    print("🌳 Creating Simple Decision Tree Visualization")
    print("=" * 50)
    
    try:
        fig = create_simple_tree_plot()
        print("\n✅ Success! The tree should now be visible in your browser.")
        print("📁 Also saved as 'simple_decision_tree.html'")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have plotly installed: pip install plotly") 