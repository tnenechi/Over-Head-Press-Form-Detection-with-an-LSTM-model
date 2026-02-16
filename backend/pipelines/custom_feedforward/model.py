import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class FeedforwardTrainer:
    def __init__(self, input_shape, hidden_sizes=[64, 32], learning_rate=0.1):  # Increased learning_rate to 0.1
        self.input_size = input_shape[1]  # 100 (features per frame)
        self.output_size = 2
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

    def initialize_weights(self):
        layers = [self.input_size] + self.hidden_sizes + [self.output_size]
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1]) * 0.01
            weights.append(w)
        return weights

    def initialize_biases(self):
        layer_sizes = self.hidden_sizes + [self.output_size]
        biases = []
        for size in layer_sizes:
            bias = np.zeros((1, size))
            biases.append(bias)
        return biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        batch_size, seq_length, _ = X.shape
        outputs = []

        # Loop through each time step
        for t in range(seq_length):
            x_t = X[:, t, :]  # Shape: (batch_size, input_size)
            activations = [x_t]

            # Loop through each layer
            for i in range(len(self.weights)):
                net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                if i < len(self.weights) - 1:
                    activation = np.maximum(0, net)  # ReLU for hidden layers
                else:
                    activation = self.sigmoid(net)  # Sigmoid for output layer
                activations.append(activation)
            outputs.append(activations[-1])
        return np.stack(outputs, axis=1)  # Shape: (batch_size, seq_length, 2)

    def backward(self, X, y, output):
        batch_size, seq_length, _ = X.shape

        # Initialize gradient lists
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        # Loop through each time step
        for t in range(seq_length):
            error = output[:, t, :] - y[:, t, :]  # Shape: (batch_size, 2)

            x_t = X[:, t, :]  # Shape: (batch_size, input_size)
            activations = [x_t]
            for i in range(len(self.weights)):
                net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                activation = np.maximum(0, net) if i < len(self.weights) - 1 else self.sigmoid(net)
                activations.append(activation)

            delta = error * (activations[-1] * (1 - activations[-1]))  # Sigmoid derivative for output layer

            for i in range(len(self.weights) - 1, -1, -1):
                dW[i] += np.dot(activations[i].T, delta)
                db[i] += np.sum(delta, axis=0, keepdims=True)
                if i > 0:
                    delta = np.dot(delta, self.weights[i].T) * (1.0 * (activations[i] > 0))  # ReLU derivative for hidden layers

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i] / seq_length
            self.biases[i] -= self.learning_rate * db[i] / seq_length

    def train(self, X, y, epochs=30, batch_size=32, validation_data=None, callbacks=None):
        history = {'accuracy': [], 'val_accuracy': []}
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))  # Ensure each epoch uses a different order to improve generalization
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)

            train_preds = self.forward(X)
            train_acc = np.mean(np.round(train_preds) == y)
            history['accuracy'].append(train_acc)

            if validation_data:
                X_val, y_val = validation_data
                val_preds = self.forward(X_val)
                val_acc = np.mean(np.round(val_preds) == y_val)
                history['val_accuracy'].append(val_acc)
            else:
                history['val_accuracy'].append(0.0)

            print(f"Epoch {epoch+1}/{epochs} - Accuracy: {train_acc:.4f}, Val Accuracy: {history['val_accuracy'][-1]:.4f}")

            if callbacks:
                for callback in callbacks:
                    callback.on_epoch_end(epoch)

        return history

    def predict(self, X):
        return self.forward(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)  # Shape: (samples, median_frames, 2)
        binary_preds = np.round(predictions)
        y_flat = y.reshape(-1, 2)  # Shape: (samples * median_frames, 2)
        binary_preds_flat = binary_preds.reshape(-1, 2)
        accuracy = np.mean(binary_preds_flat == y_flat)
        precision = precision_score(y_flat.argmax(axis=1), binary_preds_flat.argmax(axis=1), average='weighted', zero_division=0)
        recall = recall_score(y_flat.argmax(axis=1), binary_preds_flat.argmax(axis=1), average='weighted', zero_division=0)
        f1 = f1_score(y_flat.argmax(axis=1), binary_preds_flat.argmax(axis=1), average='weighted', zero_division=0)
        cm = confusion_matrix(y_flat.argmax(axis=1), binary_preds_flat.argmax(axis=1))
        print("Frame-Level Confusion Matrix:\n", cm)
        print(f"Frame-Level Accuracy: {accuracy:.4f}")
        print(f"Frame-Level Precision: {precision:.4f}")
        print(f"Frame-Level Recall: {recall:.4f}")
        print(f"Frame-Level F1-Score: {f1:.4f}")
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }

    def load_weights(self, model_path):
        try:
            data = np.load(model_path, allow_pickle=True)
            expected_keys = [f'weights_{i}' for i in range(len(self.weights))] + [f'biases_{i}' for i in range(len(self.biases))]
            if all(key in data for key in expected_keys):
                self.weights = [data[f'weights_{i}'] for i in range(len(self.weights))]
                self.biases = [data[f'biases_{i}'] for i in range(len(self.biases))]
                print(f"Loaded weights from {model_path}")
            else:
                print(f"Warning: Checkpoint {model_path} missing expected keys, using initialized weights")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {model_path}: {e}, using initialized weights")