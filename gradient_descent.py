import numpy as np
import matplotlib.pyplot as plt

def calculate_cost(points, b, m):
    error = 0

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        error += (y - (m * x + b))**2

    return error / float(len(points))

def step_gradient(points, learning_rate, b, m):
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/n) * (y - ((m * x) + b))
        m_gradient += -(2/n) * (y - ((m * x) + b)) * x

    new_b = b - (learning_rate * b_gradient)
    new_m = m - (learning_rate * m_gradient)

    return new_b, new_m

def gradient_descent_runner(points, learning_rate, b, m, num_iterations):
    for n in range(num_iterations):
        b, m = step_gradient(np.array(points), learning_rate, b, m)
#        if n % 100 == 0:
#            plot_function(points, b, m, 20, 60)

    return b, m

def plot_function(points, b, m, x1, x2):
    plt.xlabel('Flaeche in qm')
    plt.ylabel('Kosten/Monat in Euro')
    plt.plot(points[:,0], points[:,1], 'o')
    y1 = m * x1 + b
    y2 = m * x2 + b
    x = [x1, y1]
    y = [x2, y2]
    plt.plot(x,y)
    plt.show()

def main():
    # y = m * x + b
    # x = parameters
    points = np.genfromtxt('/home/mh/Programs/Python/ML/GradientDescent/data2.csv', delimiter=',') # x
    learning_rate = 0.0001
    b = 0
    m = 0
    num_iterations = 1000

    b, m = gradient_descent_runner(points, learning_rate, b, m, num_iterations)
    print 'After {} iterations with a learning rate of {}: b = {}, m = {}'.format(num_iterations, learning_rate, b, m)

if __name__ == '__main__':
    main()
