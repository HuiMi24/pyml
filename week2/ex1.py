import numpy as np
import matplotlib.pylab as plt

def warmUpExercise():
    return np.identity(5)

def load(file_name):
    X, y = [], []
    with open(file_name) as fd:
        for line in fd.readlines():
            liner = line.strip().split(',')
            X.append([1.0, liner[0]])
            y.append([liner[1]])
    return np.array(X, dtype='float'), np.array(y, dtype='float')

def plotData(X, y):
    plt.plot(X, y, 'r.')
    plt.show()

def computeCost(X, y, theta):
    J = 0
    m, _ = y.shape
    predictions = np.dot(X, theta)
    J = sum((predictions - y)**2)/(2.0 * m)
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m, _ = y.shape
    J_history = np.zeros((iterations, 1))
    for i in range(iterations):
        predictions = np.dot(X, theta)
        updates = np.dot(X.T, predictions - y)
        theta = theta - alpha * (1.0/m) * updates
        J_history[i][0] = computeCost(X, y, theta)
    return J_history, theta

print("Running warmUpExercise ... ")
print("5x5 Identity Matrix: ")
print warmUpExercise()

print("Plotting Data ...")
X, y = load('ex1data1.txt')

plotData(X[:,1], y)

m, _ = y.shape

theta = np.zeros((2,1))
J = computeCost(X, y, theta)

print("With theta = [0 ; 0]\nCost computed = %f" % (J))
print("Expected cost value (approx) 32.07")

J = computeCost(X, y, np.array([[-1], [2]]))
print("With theta = [-1 ; 2]\nCost computed = %f" % (J))
print("Expected cost value (approx) 54.24")

print("Running Gradient Descent ...")

iterations = 1500
alpha = 0.01
J_history, theta = gradientDescent(X, y, theta,alpha, iterations)
print("Theta found by gradient descent:")
print(theta)
print("Expected theta values (approx)")
print(" -3.6303\n  1.1664")

plt.plot(J_history)
plt.ylabel('Cost J')
plt.xlabel('Iterations');
plt.show()


plt.plot(X[:, 1], y, 'rx')
plt.plot(X[:, 1], np.dot(X, theta), '-')
plt.show()

print("For population = 35,000, we predict a profit of %f" % (np.dot([1, 3.5], theta) * 10000))
print("For population = 70,000, we predict a profit of %f" % (np.dot([1, 7], theta) * 10000))

