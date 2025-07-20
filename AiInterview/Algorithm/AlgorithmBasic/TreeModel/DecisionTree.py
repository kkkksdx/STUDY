import numpy as np
import pandas as pd
from typing import Union, Tuple

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, task_type='classification'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        if(task_type not in ['classification', 'regression']):
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.task_type = task_type
        self.classes = None
    
    class Node:
        def __init__(self):
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.value = None
            self.is_leaf = False
    
    def _caculate_mse(self,y:np.ndarray) -> float:
        return np.mean((y-np.mean(y))**2)

    def _caculate_gini(self,y:np.ndarray) -> float:
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)


    def _caculate_impurity(self,y:np.ndarray) -> float:
        if self.task_type == 'regression':
            return self._caculate_mse(y)
        else:
            return self._caculate_gini(y)

    def _find_best_split(self, X:np.ndarray, y:np.ndarray) -> Tuple[int, float, float]:
        best_feature = None
        best_threshold = None
        best_score = float('inf')
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_impurity = self._caculate_impurity(y[left_mask])
                right_impurity = self._caculate_impurity(y[right_mask])

                current_score = np.sum(left_mask) * left_impurity + np.sum(right_mask) * right_impurity
                if current_score < best_score:
                    best_score = current_score
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_score
                
    
    def _bulid_tree(self, X:np.ndarray, y:np.ndarray, depth:int) -> Node:
        node = self.Node()
        if(self.max_depth is not None and depth >= self.max_depth) or \
            (len(y) < self.min_samples_split) or \
            (len(np.unique(y)) == 1):
            node.is_leaf = True
            if self.task_type == 'regression':
                node.value = np.mean(y)
            else:
                node.value = self.classes[np.argmax(np.bincount(y))]
            return node
        
        best_feature, threshold, score = self._find_best_split(X, y)
        if best_feature is None:
            node.is_leaf = True
            node.value = np.mean(y) 
            return node  
        
        node.feature = best_feature
        node.threshold = threshold
        node.left = self._build_tree(X[X[:, best_feature] <= threshold], y[X[:, best_feature] <= threshold], depth+1)
        node.right = self._build_tree(X[X[:, best_feature] > threshold], y[X[:, best_feature] > threshold], depth+1)
        return node
        
    def fit(self, X:Union[np.ndarray, pd.DataFrame], y:Union[np.ndarray, pd.Series]):
        # 训练决策树
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if self.task_type == 'classification':
            self.classes = np.unique(y)
            y = np.searchsorted(self.classes, y)
        
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _predict(self, x:np.ndarray, node:Node) -> Union[float, str]:
        if node.is_leaf:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)


    def predict(self, X):
        # 预测新数据
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._predict(x, self.tree) for x in X])
        
        
        
