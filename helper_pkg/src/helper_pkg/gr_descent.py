import numpy as np

def computeCost( X, y, theta):
    m = (X.T @ theta.T)
    inner = np.power(((X @ theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y):
    theta = np.array([[1.0, 1.0]])
    iters = 10
    alpha = 0.001
    cost = 0
    for i in range(iters):
        theta = theta - (alpha / len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
        cost = computeCost(X, y, theta)
        # if i % 10 == 0: # just look at cost every ten loops for debugging
        #     print(cost)
    return (theta, cost)

X = np.array([2.0,3.0]).reshape(-1,1)
ones = np.ones([X.shape[0], 1])
X = np.concatenate([ones, X],1)
y = np.array([3.0,0.2]).reshape(-1,1)
print(gradientDescent(X,y))