import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, TimeDistributed, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
from tensorflow.keras.callbacks import Callback

class SaveHistoryCallback(Callback):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        history_path = os.path.join(self.checkpoint_dir, f"history_epoch_{epoch + 1}.npy")
        np.save(history_path, logs)



class LSTMTrainer:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)

    def build_inception_module(self, inputs, units):
        branch1 = LSTM(units // 4, return_sequences=True)(inputs)
        branch2 = LSTM(units // 2, return_sequences=True)(inputs)
        branch3 = LSTM(units, return_sequences=True)(inputs)
        return Concatenate()([branch1, branch2, branch3])

    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        
        x = self.build_inception_module(inputs, 64)
        x = Dropout(0.2)(x)
        x1 = x
        
        x = self.build_inception_module(x, 128)
        x = Dropout(0.2)(x)
        x2 = x
        
        x = self.build_inception_module(x, 512)
        x = Dropout(0.2)(x)
        
        x = self.build_inception_module(x, 128)
        x = Dropout(0.2)(x)
        x = Add()([x, x2])
        
        x = self.build_inception_module(x, 64)
        x = Dropout(0.2)(x)
        x = Add()([x, x1])
        
        outputs = TimeDistributed(Dense(2, activation='sigmoid'))(x)  # Frame-level predictions
        model = Model(inputs, outputs)
        return model

    def train(self, x, y=None, validation_data=None, epochs=50, batch_size=32, callbacks=None, initial_epoch=0):
        self.model.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
        
        checkpoint_dir = '/content/drive/MyDrive/backend/pipelines/lstm/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        history_callback = SaveHistoryCallback(checkpoint_dir)
        all_callbacks = [history_callback] + (callbacks if callbacks else [])


        history = self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=1,
            callbacks=all_callbacks,
            initial_epoch=initial_epoch
        )
        return history.history

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)  # Shape: (samples, median_frames, 2)
        y_pred_binary = (y_pred >= 0.5).astype(np.int8)  # Binarize predictions
        
        # Flatten for per-frame evaluation
        y_test_flat = y_test.reshape(-1, 2)
        y_pred_flat = y_pred_binary.reshape(-1, 2)
        
        # Compute metrics for knees and elbows separately
        cm_knees = confusion_matrix(y_test_flat[:, 0], y_pred_flat[:, 0])
        cm_elbows = confusion_matrix(y_test_flat[:, 1], y_pred_flat[:, 1])
        accuracy = accuracy_score(y_test_flat.argmax(axis=1), y_pred_flat.argmax(axis=1))
        precision = precision_score(y_test_flat.argmax(axis=1), y_pred_flat.argmax(axis=1), average='weighted', zero_division=0)
        recall = recall_score(y_test_flat.argmax(axis=1), y_pred_flat.argmax(axis=1), average='weighted', zero_division=0)
        f1 = f1_score(y_test_flat.argmax(axis=1), y_pred_flat.argmax(axis=1), average='weighted', zero_division=0)
        
        print("Frame-Level Confusion Matrix - Knees:\n", cm_knees)
        print("Frame-Level Confusion Matrix - Elbows:\n", cm_elbows)
        print(f"Frame-Level Accuracy: {accuracy:.4f}")
        print(f"Frame-Level Precision: {precision:.4f}")
        print(f"Frame-Level Recall: {recall:.4f}")
        print(f"Frame-Level F1-Score: {f1:.4f}")
        return {
            'confusion_matrix_knees': cm_knees.tolist(),
            'confusion_matrix_elbows': cm_elbows.tolist(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }