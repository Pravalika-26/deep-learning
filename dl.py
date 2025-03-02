import deeplake
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load SVHN dataset from Activeloop Deep Lake
print("Loading SVHN dataset...")
train_ds = deeplake.load('hub://activeloop/svhn-train')
test_ds = deeplake.load('hub://activeloop/svhn-test')

# Function to preprocess the dataset
def preprocess_svhn(dataset):
    images = [sample.numpy() for sample in dataset['images']]
    labels = [sample.numpy() for sample in dataset['labels']]

    images = np.array(images, dtype=np.float32) / 255.0  # Normalize
    labels = np.array(labels, dtype=np.int32)

    labels = tf.keras.utils.to_categorical(labels, num_classes=10)  # One-hot encode
    return images.reshape(images.shape[0], -1), labels  # Flatten images


# Prepare training and testing data
X_train, y_train = preprocess_svhn(train_ds)
X_test, y_test = preprocess_svhn(test_ds)

# Split training set into train (90%) and validation (10%)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print(f"Training Data Shape: {X_train.shape}")
print(f"Validation Data Shape: {X_val.shape}")
print(f"Testing Data Shape: {X_test.shape}")

# Define the Feedforward Neural Network
class FeedForwardNN:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', weight_init='random'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_func = self.relu if activation == 'relu' else self.sigmoid

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            if weight_init == 'xavier':
                limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
                W = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            else:
                W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            self.weights.append(W)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights) - 1):
            X = self.activation_func(np.dot(X, self.weights[i]) + self.biases[i])
            self.activations.append(X)

        X = self.softmax(np.dot(X, self.weights[-1]) + self.biases[-1])
        self.activations.append(X)
        return X

    def backward(self, X, Y, output, learning_rate):
        m = X.shape[0]
        dZ = output - Y
        gradients = []

        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            gradients.insert(0, (dW, db))
            dZ = np.dot(dZ, self.weights[i].T) * (self.activations[i] > 0)  # ReLU derivative

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

    def train(self, X, Y, epochs=10, learning_rate=0.001, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]

                output = self.forward(X_batch)
                self.backward(X_batch, Y_batch, output, learning_rate)

            if epoch % 2 == 0:
                loss = -np.sum(Y * np.log(output + 1e-9)) / X.shape[0]
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Model hyperparameters
hidden_layers = [64, 128, 64]
learning_rate = 0.001
epochs = 10
batch_size = 32

# Train the model
print("Training the model...")
model = FeedForwardNN(input_size=X_train.shape[1], hidden_sizes=hidden_layers, output_size=10, activation='relu', weight_init='xavier')
model.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

# Evaluate the model
print("Evaluating the model...")
predictions = model.predict(X_test)
accuracy = np.mean(np.argmax(y_test, axis=1) == predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# Compute confusion matrix
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), predictions)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
