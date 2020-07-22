import numpy as np
from math import sqrt
from collections import Counter

def kNN_classify(k,X_train,y_train,x):
    
    # 断言(必须满足的条件)
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], "the rows of X_train must be equal to that of y_train"
    # 对于单实例数据，需要使用x.reshape(1,-1)
    # 对于单特征数据，需要使用x.reshape(-1,1)
    # 否则，assert X_train.shape[1] == x.shape[1]报错
    assert X_train.shape[1] == x.shape[1], "the features number of x must be equal to that of X_train"
    
    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    nearest = np.argsort(distances);
    
    topK_y = [y_train[neighbor] for neighbor in nearest[:k]];
    
    votes = Counter(topK_y);
    
    return votes.most_common(1)[0][0]
    