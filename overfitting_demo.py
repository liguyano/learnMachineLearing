import numpy as np
import matplotlib.pyplot as plt
from handwritten_decision_tree import HandwrittenDecisionTree
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class OverfittingDemo:
    def __init__(self):
        self.results = []
    
    def demonstrate_overfitting(self):
        """Show how overfitting occurs with different tree depths"""
        print("🌳 OVERFITTING DEMONSTRATION")
        print("=" * 50)
        
        # Load data
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Test different max_depths
        depths = range(1, 15)
        train_accuracies = []
        test_accuracies = []
        tree_sizes = []
        
        print("Testing different tree depths...")
        print("Depth | Train Acc | Test Acc | Tree Size | Status")
        print("-" * 55)
        
        for depth in depths:
            # Train tree with specific depth
            dt = HandwrittenDecisionTree(
                max_depth=depth,
                min_samples_split=2,
                min_samples_leaf=1
            )
            dt.fit(X_train, y_train)
            
            # Calculate accuracies
            train_pred = dt.predict(X_train)
            test_pred = dt.predict(X_test)
            
            train_acc = np.mean(train_pred == y_train)
            test_acc = np.mean(test_pred == y_test)
            
            # Get tree info
            tree_depth, leaves, nodes = dt.get_tree_info()
            
            # Determine status
            if train_acc - test_acc > 0.1:
                status = "🚨 OVERFITTING"
            elif train_acc - test_acc > 0.05:
                status = "⚠️  SLIGHT OVERFIT"
            else:
                status = "✅ GOOD"
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            tree_sizes.append(nodes)
            
            print(f"{depth:5d} | {train_acc:8.3f} | {test_acc:7.3f} | {nodes:8d} | {status}")
        
        # Plot the results
        self.plot_overfitting_curve(depths, train_accuracies, test_accuracies, tree_sizes)
        
        return depths, train_accuracies, test_accuracies
    
    def plot_overfitting_curve(self, depths, train_acc, test_acc, tree_sizes):
        """Plot training vs testing accuracy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy plot
        ax1.plot(depths, train_acc, 'o-', color='blue', label='Training Accuracy', linewidth=2)
        ax1.plot(depths, test_acc, 'o-', color='red', label='Testing Accuracy', linewidth=2)
        ax1.set_xlabel('Tree Depth')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Overfitting: Training vs Testing Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add annotations
        max_gap_idx = np.argmax(np.array(train_acc) - np.array(test_acc))
        ax1.annotate('🚨 Maximum Overfitting', 
                    xy=(depths[max_gap_idx], test_acc[max_gap_idx]),
                    xytext=(depths[max_gap_idx] + 2, test_acc[max_gap_idx] - 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        # Tree complexity plot
        ax2.bar(depths, tree_sizes, color='green', alpha=0.7)
        ax2.set_xlabel('Tree Depth')
        ax2.set_ylabel('Number of Nodes')
        ax2.set_title('Tree Complexity vs Depth')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_regularization_techniques(self):
        """Show different techniques to prevent overfitting"""
        print("\n🛡️ OVERFITTING PREVENTION TECHNIQUES")
        print("=" * 50)
        
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        techniques = [
            {
                'name': '🚫 No Regularization',
                'params': {'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1}
            },
            {
                'name': '📏 Max Depth Limit',
                'params': {'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 1}
            },
            {
                'name': '📊 Min Samples Split',
                'params': {'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 1}
            },
            {
                'name': '🍃 Min Samples Leaf',
                'params': {'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 5}
            },
            {
                'name': '🛡️ Combined Regularization',
                'params': {'max_depth': 5, 'min_samples_split': 8, 'min_samples_leaf': 3}
            }
        ]
        
        print("Technique                  | Train Acc | Test Acc | Gap   | Nodes | Status")
        print("-" * 80)
        
        results = []
        for technique in techniques:
            dt = HandwrittenDecisionTree(**technique['params'])
            dt.fit(X_train, y_train)
            
            train_pred = dt.predict(X_train)
            test_pred = dt.predict(X_test)
            
            train_acc = np.mean(train_pred == y_train)
            test_acc = np.mean(test_pred == y_test)
            gap = train_acc - test_acc
            
            _, _, nodes = dt.get_tree_info()
            
            if gap > 0.1:
                status = "🚨 OVERFITTING"
            elif gap > 0.05:
                status = "⚠️  SLIGHT OVERFIT"
            else:
                status = "✅ WELL REGULARIZED"
            
            results.append({
                'name': technique['name'],
                'train_acc': train_acc,
                'test_acc': test_acc,
                'gap': gap,
                'nodes': nodes,
                'status': status
            })
            
            print(f"{technique['name']:25s} | {train_acc:8.3f} | {test_acc:7.3f} | {gap:4.3f} | {nodes:4d} | {status}")
        
        return results
    
    def create_synthetic_overfitting_example(self):
        """Create a synthetic dataset to clearly show overfitting"""
        print("\n🧪 SYNTHETIC OVERFITTING EXAMPLE")
        print("=" * 50)
        
        # Create a small, noisy dataset
        np.random.seed(42)
        X, y = make_classification(
            n_samples=100,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Add some noise
        noise = np.random.normal(0, 0.1, X.shape)
        X_noisy = X + noise
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_noisy, y, test_size=0.4, random_state=42
        )
        
        print(f"Dataset: {len(X_train)} training samples, {len(X_test)} test samples")
        
        # Compare different approaches
        approaches = [
            ('Overfitted Tree', {'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1}),
            ('Regularized Tree', {'max_depth': 4, 'min_samples_split': 8, 'min_samples_leaf': 4}),
        ]
        
        for name, params in approaches:
            dt = HandwrittenDecisionTree(**params)
            dt.fit(X_train, y_train)
            
            train_acc = np.mean(dt.predict(X_train) == y_train)
            test_acc = np.mean(dt.predict(X_test) == y_test)
            depth, leaves, nodes = dt.get_tree_info()
            
            print(f"\n{name}:")
            print(f"  Training Accuracy: {train_acc:.3f}")
            print(f"  Testing Accuracy:  {test_acc:.3f}")
            print(f"  Overfitting Gap:   {train_acc - test_acc:.3f}")
            print(f"  Tree Complexity:   {nodes} nodes, {leaves} leaves, depth {depth}")
    
    def show_best_practices(self):
        """Show best practices for preventing overfitting"""
        print("\n🎯 BEST PRACTICES TO PREVENT OVERFITTING")
        print("=" * 50)
        
        practices = [
            "📏 **Limit Tree Depth**: Set max_depth=3-7 for most problems",
            "📊 **Minimum Sample Split**: Require min_samples_split=5-20",
            "🍃 **Minimum Leaf Size**: Set min_samples_leaf=3-10",
            "✂️ **Post Pruning**: Remove branches that don't improve validation accuracy",
            "🎲 **Cross Validation**: Use k-fold CV to select best parameters",
            "🌲 **Ensemble Methods**: Use Random Forest or Gradient Boosting",
            "📈 **Learning Curves**: Plot train/test accuracy vs tree size",
            "🎯 **Early Stopping**: Stop when validation accuracy stops improving",
            "📊 **More Data**: Collect more training samples if possible",
            "🔍 **Feature Selection**: Remove irrelevant or noisy features"
        ]
        
        for i, practice in enumerate(practices, 1):
            print(f"{i:2d}. {practice}")
        
        print(f"\n💡 **Rule of Thumb**: If training accuracy >> test accuracy, you're overfitting!")

def main():
    demo = OverfittingDemo()
    
    # Run all demonstrations
    demo.demonstrate_overfitting()
    demo.demonstrate_regularization_techniques()
    demo.create_synthetic_overfitting_example()
    demo.show_best_practices()
    
    print("\n" + "=" * 60)
    print("🎓 KEY TAKEAWAY:")
    print("Simple trees generalize better than complex ones!")
    print("Always validate on unseen data to detect overfitting!")
    print("=" * 60)

if __name__ == "__main__":
    main() 