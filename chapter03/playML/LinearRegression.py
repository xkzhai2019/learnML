import numpy as np
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None # 下划线开头，用作私有变量

    def fit_normal(self, X_train, y_train):
        """正规方程解"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train]) # 训练数据集中添加常数列
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self
    
    def fit_gd(self, X_train, y_train, eta = 0.01, n_iters = 1e4):
        """梯度下降法求解线性回归"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        def J(theta,X_b,y):
            try:
                return np.sum((y - X_b.dot(theta))**2) / len(y)
            except:
                return float('inf')
            
        def dJ(theta, X_b, y):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
            return res * 2 / len(X_b)
        
        def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon=1e-8):
    
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
            
                cur_iter += 1

            return theta
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train]) # 训练数据集中添加常数列
        initial_theta = np.zeros((X_b.shape[1]))
        
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters);

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
        return self

    def fit_gd_vec(self, X_train, y_train, eta = 0.01, n_iters = 1e4):
        """梯度下降法求解线性回归,向量化"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        def J(theta,X_b,y):
            try:
                return np.sum((y - X_b.dot(theta))**2) / len(y)
            except:
                return float('inf')
            
        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)
        
        def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon=1e-8):
    
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
            
                cur_iter += 1

            return theta
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train]) # 训练数据集中添加常数列
        initial_theta = np.zeros((X_b.shape[1]))
        
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters);

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
        return self

    def fit_sgd(self, X_train, y_train, n_iters = 5, a = 5, b = 50):
        """随机梯度下降法求解线性回归"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1
            
        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.
        
        def sgd(X_b, y, initial_theta, n_iters=5, a = 5, b = 50):

            # 定义逐步递减的学习率
            def learning_rate(t):
                return a / (t + b)    
            
            theta = initial_theta
            m = len(X_b)
            
# 并不能保证真的能把数据集中每个样本都看n_iters遍
#             for cur_iter in range(n_iters * m);
#                 rand_i = np.random.randin(m)
#                 gradient = dJ_sgd(theta, X_b[rand_i], y[rand_i])
#                 theta = theta - learning_rate(cur_iter) * gradient    
            
            for i_iter in range(n_iters):
                # 在每次迭代中，将训练集中的所有数据都看一遍
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes,:]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(i_iter * m + i) * gradient                

            return theta
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train]) # 训练数据集中添加常数列
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, a , b);

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
        return self
    
    def fit_mini_sgd(self, X_train, y_train, k=32, n_iters = 5, a = 5, b = 50):
        """小批量随机梯度下降法求解线性回归"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1
        
            
        def dJ_min_sgd(theta, X_b_k, y_k):
            return 2 * X_b_k.T.dot(X_b_k.dot(theta) - y_k) / len(y_k)
        
        def mini_sgd(X_b, y, initial_theta, k=32, n_iters=5, a = 5, b = 50):

            # 定义逐步递减的学习率
            def learning_rate(t):
                return a / (t + b)    
            
            theta = initial_theta
            m = len(X_b)
            
            # 看完整个数据集需要的次数
            iters = m // k
            # 剩余样本数
            left_num = m % k
            
            # 实际更新次数
            times = 0;
            
            
#          cur_iter = 0
#          while cur_iter < n_iters:
#               # 每次迭代随机选取k个样本，计算方向
#               indexes = np.random.permutation(m)
#               X_b_k = X_b[:k,:]
#               y_k = y[:k]
#               gradient = dJ_mini_gd(theta, X_b_k, y_k)
#               theta = theta - eta * gradient

#               cur_iter += 1  
            
            for i_iter in range(n_iters):
                # 在每次迭代中，将训练集中的所有数据都看一遍
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes,:]
                y_new = y[indexes]
                
                for i in range(iters):
                    X_b_k = X_b_new[(i*k):((i+1)*k)]
                    y_k = y_new[(i*k):((i+1)*k)]
                    gradient = dJ_min_sgd(theta, X_b_k, y_k)
                    theta = theta - learning_rate(times) * gradient
                    times += 1
                if left_num>0:
                    X_b_left = X_b_new[((i+1)*k):]
                    y_left = y_new[((i+1)*k):]
                    gradient = dJ_min_sgd(theta, X_b_left, y_left)
                    theta = theta - learning_rate(times) * gradient
                    times += 1
                    
            return theta
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train]) # 训练数据集中添加常数列
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta = mini_sgd(X_b, y_train, initial_theta, k, n_iters, a, b);

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
        return self    
    

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict]) # 数据集中添加常数列
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的性能"""

        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
