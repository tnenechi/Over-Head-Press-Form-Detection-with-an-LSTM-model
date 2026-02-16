# model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class CustomLSTM:
    def __init__(self, input_size, hidden_sizes, output_size, seq_length, lr=0.001, seed=42):
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.seq_length = seq_length
        self.lr = lr
        self._keras_model = None

        # weights for forward pass
        self.Wx, self.Wh, self.b = [], [], []
        for idx, H in enumerate(hidden_sizes):
            in_dim = input_size if idx == 0 else hidden_sizes[idx-1]    #set input dimension; first layer = input size; hidden layers = previous layer's hidden size

            self.Wx.append(np.random.randn(in_dim, 4*H) / np.sqrt(in_dim))  # input weight matrix
            self.Wh.append(np.random.randn(H, 4*H) / np.sqrt(H))    # recurrent weight matrix
            self.b.append(np.zeros((1, 4*H)))   # bias vector
        self.W_out = np.random.randn(hidden_sizes[-1], output_size) / np.sqrt(hidden_sizes[-1]) # output weight matrix
        self.b_out = np.zeros((1, output_size)) #output bias vector

    # Forward pass
    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        h = [np.zeros((batch_size, H)) for H in self.hidden_sizes]  # hidden states for each LSTM layer
        c = [np.zeros((batch_size, H)) for H in self.hidden_sizes]  # cell states 
        outputs = []

        # Loop through time steps
        for t in range(seq_len):
            x_t = X[:, t, :]
            for l, H in enumerate(self.hidden_sizes):
                a = x_t @ self.Wx[l] + h[l] @ self.Wh[l] + self.b[l] # pre-activation: shape(batch_size, 4H)
                ai, af, ag, ao = np.split(a, 4, axis=1) # 4 gates each of shape(batch_size, H)

                i = 1 / (1 + np.exp(-ai))   #input gate; sigmoid function
                f = 1 / (1 + np.exp(-af))   #forget gate; sigmoid function
                g = np.tanh(ag)             #cell gate
                o = 1 / (1 + np.exp(-ao))   #output gate

                c_t = f * c[l] + i * g  #update cell state
                h_t = o * np.tanh(c_t)  #new hidden state
                x_t, h[l], c[l] = h_t, h_t, c_t #updates for next layer
            y_hat = 1 / (1 + np.exp(-(x_t @ self.W_out + self.b_out)))
            outputs.append(y_hat)
        return np.stack(outputs, axis=1)


    # Keras model 
    def build_keras_model(self):
        input_seq = Input(shape=(self.seq_length, self.input_size))
        x = input_seq   # input layer

        # Loop over hidden layers
        for H in self.hidden_sizes:
            x = LSTM(H, return_sequences=True)(x)
        output = TimeDistributed(Dense(self.output_size, activation='sigmoid'))(x)

        # configure the model
        model = Model(inputs=input_seq, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self._keras_model = model
        return model

    def train_keras(self, X_train, y_train, X_val=None, y_val=None,
                    epochs=50, batch_size=16, callbacks=None):
        if self._keras_model is None:
            self.build_keras_model()
        history = self._keras_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict(self, X):
        if self._keras_model is None:
            raise ValueError("Keras model not built yet. Call build_keras_model() first.")
        return self._keras_model.predict(X)

    # Metrics evaluation 
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred_binary = (y_pred >= 0.5).astype(np.int8)

        # Flatten for per-frame evaluation
        y_test_flat = y_test.reshape(-1, 2)
        y_pred_flat = y_pred_binary.reshape(-1, 2)

        cm_knees = confusion_matrix(y_test_flat[:, 0], y_pred_flat[:, 0])
        cm_elbows = confusion_matrix(y_test_flat[:, 1], y_pred_flat[:, 1])
        accuracy = accuracy_score(y_test_flat.argmax(axis=1), y_pred_flat.argmax(axis=1))
        precision = precision_score(y_test_flat.argmax(axis=1), y_pred_flat.argmax(axis=1), average='weighted', zero_division=0)
        recall = recall_score(y_test_flat.argmax(axis=1), y_pred_flat.argmax(axis=1), average='weighted', zero_division=0)
        f1 = f1_score(y_test_flat.argmax(axis=1), y_pred_flat.argmax(axis=1), average='weighted', zero_division=0)

        print("Confusion Matrix - Knees:\n", cm_knees)
        print("Confusion Matrix - Elbows:\n", cm_elbows)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        return {
            "confusion_matrix_knees": cm_knees.tolist(),
            "confusion_matrix_elbows": cm_elbows.tolist(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def save_keras_weights(self, filepath):
        if self._keras_model:
            self._keras_model.save(filepath)

    def load_keras_weights(self, filepath):
        self._keras_model = tf.keras.models.load_model(filepath)


    # Save & Load NumPy Weights 
    def save_weights(self, filepath):
        """Save weights of Keras model into NumPy .npz for forward pass usage."""
        if self._keras_model is None:
            raise ValueError("Keras model not built yet.")

        weights = {}
        for i, layer in enumerate(self._keras_model.layers):
            for j, w in enumerate(layer.get_weights()):
                weights[f"layer_{i}_w_{j}"] = w
        np.savez(filepath, **weights)

    def load_weights(self, filepath):
        """Load weights from .npz into NumPy dict (for manual forward pass)."""
        data = np.load(filepath)
        return {k: data[k] for k in data}
