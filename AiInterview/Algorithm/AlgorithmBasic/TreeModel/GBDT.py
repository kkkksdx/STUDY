import numpy as np
from DecisionTree import DecisionTree

class GBDT:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        """
        n_estimators: 树的个数
        learning_rate: 学习率
        max_depth: 树的深度
        min_samples_split: 最小分割样本数
        """
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        """
        训练模型
        X: 特征矩阵
        y: 目标变量
        """
        # 初始化残差
        F = np.zeros_like(y, dtype=np.float64)
        # 训练树
        for _ in range(self.n_estimators):
            # 计算残差
            residuals = y - F
            # 训练决策树
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,task_type='regression')
            tree.fit(X, residuals)
            # 更新模型
            F += self.learning_rate * tree.predict(X)
            # 保存树
            self.trees.append(tree)
        return self

    def predict(self, X):
        predictions = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

    def score(self, X, y):
        """
        计算R^2得分
        """
        y_pred = self.predict(X)
        score_total = np.sum((y- np.mean(y))**2)
        score_residual = np.sum((y-y_pred)**2)
        return 1 - score_residual / score_total
        