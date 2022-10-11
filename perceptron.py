import numpy as np


class Perceptron:
    """
    Simple perceptron class using unit step activation
    """
    def __init__(self, l_rate=1.0, n_iterations=1000):
        self.l_rate = l_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def predict(self, x_i):
        activation = np.dot(x_i, self.weights) + self.bias
        return np.where(activation >= 0, 1, 0)

    def fit(self, x, y):

        # initialize weights and bias
        self.weights = np.zeros(len(x[0]))
        self.bias = 0

        for _ in range(self.n_iterations):
            for index, x_i in enumerate(x):
                prediction = self.predict(x_i)

                update = self.l_rate * (y[index] - prediction)

                self.weights += update * x_i
                self.bias += update


def main():
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron(l_rate=1.0, n_iterations=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()


if __name__ == '__main__':
    main()
