import numpy as np

def load(file_name):
    X, y = [], []
    with open(file_name) as fd:
        for line in fd.readlines():
            liner = line.strip().split(',')
            X.append([1.0, liner[0], liner[1]])
            y.append([liner[2]])
    return np.array(X, dtype='float'), np.array(y, dtype='float')

def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    
    n = X.shape[1]
    for i in range(n):
        mu[0, i] = np.mean(X[:, i])
        sigma[0, i] = np.std(X[:, i])
        X_norm[:,i] = (X[:, i] - mu[0,i])/sigma[0, i]
    return X_norm, mu, sigma

X, y = load('ex1data2.txt')
m = y.shape[0]

print("Normalizing Features ...")
X,mu,sigma = featureNormalize(X[:, 1:]);
print X