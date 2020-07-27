import numpy as np

class SimpleLinearRegression1:
    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        self.a_ = None
        self.b_ = None    
    
    def fit(self, X_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        assert len(X_train) == len(y_train),\
            "the size of X_train must be equal to the size of y_train."
        assert X_train.ndim == 1,\
            "Simple Linear Regressor only solve signgle feature training data."
        
        X_mean = np.mean(X_train);
        y_mean = np.mean(y_train);
        
        num = 0.0 # 分子
        d  = 0.0 # 分母

        # for 循环计算a,b
        for x_i, y_i in zip(X_train,y_train):
            num += (x_i - X_mean) * (y_i - y_mean)
            d += (x_i - X_mean) ** 2
        
        self.a_ = num/d;
        self.b_ = y_mean - self.a_ * X_mean;
        
        return self
    
    def predict(self, X_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert X_predict.ndim == 1,\
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None,\
            "must fit before predict!"
        
        return np.array([self._predict(x) for x in X_predict])
    
    def _predict(self,x_single):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"
    

class SimpleLinearRegression2:
    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        self.a_ = None
        self.b_ = None    
    
    def fit(self, X_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        assert len(X_train) == len(y_train),\
            "the size of X_train must be equal to the size of y_train."
        assert X_train.ndim == 1,\
            "Simple Linear Regressor only solve signgle feature training data."
        
        X_mean = np.mean(X_train);
        y_mean = np.mean(y_train);

        # 向量化计算a,b
        num = (X_train - X_mean).dot(y_train - y_mean);
        d = (X_train - X_mean).dot(X_train - X_mean);
        
        self.a_ = num/d;
        self.b_ = y_mean - self.a_ * X_mean;
        
        return self
    
    def predict(self, X_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert X_predict.ndim == 1,\
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None,\
            "must fit before predict!"
        
        return np.array([self._predict(x) for x in X_predict])
    
    def _predict(self,x_single):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression2()"