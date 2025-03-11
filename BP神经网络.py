import numpy as np
import matplotlib.pyplot as plt

# Normalize input data
def normalize(x):
    return (x - np.mean(x)) / np.std(x)

x = np.linspace(-1, 1, 100).reshape(-1, 1)
y = x**2 + 0.2 * np.random.rand(100, 1)

x_normalized = normalize(x)
y_normalized = normalize(y)

class Net:
    def __init__(self, n_features, n_hidden, n_output):
        self.hidden_weights = np.random.randn(n_features, n_hidden) * np.sqrt(2 / n_features)  # Xavier initialization
        self.hidden_bias = np.zeros((1, n_hidden))
        self.output_weights = np.random.randn(n_hidden, n_output) * np.sqrt(2 / n_hidden)  # Xavier initialization
        self.output_bias = np.zeros((1, n_output))

    def forward(self, x):
        self.hidden_layer = np.maximum(0, np.dot(x, self.hidden_weights) + self.hidden_bias)
        output_layer = np.dot(self.hidden_layer, self.output_weights) + self.output_bias
        return output_layer

net = Net(1, 10, 1)

print(net)

plt.ion()
plt.show()

learning_rate = 0.01

for t in range(1000):  # Increased number of iterations for better convergence
    prediction = net.forward(x_normalized)
    loss = np.mean(np.square(prediction - y_normalized))

    # Gradient descent
    output_error = 2 * (prediction - y_normalized) / len(y_normalized)
    output_delta = np.dot(output_error, net.output_weights.T)
    output_weights_update = np.dot(net.hidden_layer.T, output_error)
    net.output_weights -= learning_rate * output_weights_update
    net.output_bias -= learning_rate * np.sum(output_error, axis=0, keepdims=True)

    hidden_error = output_delta * (net.hidden_layer > 0)
    hidden_delta = np.dot(hidden_error, net.hidden_weights.T)
    hidden_weights_update = np.dot(x_normalized.T, hidden_error)
    net.hidden_weights -= learning_rate * hidden_weights_update
    net.hidden_bias -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

    if t % 100 == 0:
        plt.cla()
        plt.scatter(x_normalized, y_normalized)
        plt.plot(x_normalized, prediction, 'r-', lw=2)
        plt.text(0.5, -1, f'loss={loss:.4f}', fontdict={'size': 12, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()