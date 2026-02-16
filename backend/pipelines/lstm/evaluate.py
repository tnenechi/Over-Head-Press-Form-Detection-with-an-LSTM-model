import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from utils.preprocess import Preprocessor
from pipelines.lstm.model import LSTMTrainer


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
data_dir = os.path.join(BASE_DIR, "data")
checkpoint_dir = os.path.join(BASE_DIR, "pipelines", "lstm", "checkpoints") 
metrics_dir = os.path.join(BASE_DIR, "pipelines", "lstm", "metrics")

# Initialize preprocessor
preprocessor = Preprocessor(data_dir)  
input_shape = (preprocessor.median_frames, 100)

# Load checkpoint
checkpoint_file = os.path.join(checkpoint_dir, "lstm_model.h5")
if not os.path.exists(checkpoint_file):
    raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}")
print(f"Loading model from: {checkpoint_file}")

trainer = LSTMTrainer(input_shape)
trainer.model = load_model(checkpoint_file)

# Create test dataset
def create_dataset(keys):
    def generator():
        for X_batch, y_batch in preprocessor.load_data(keys, batch_size=32):
            yield X_batch, y_batch
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(tf.float32, tf.int8),
        output_shapes=([None, preprocessor.median_frames, 100], [None, preprocessor.median_frames, 2])
    ).prefetch(tf.data.AUTOTUNE)
    return dataset

test_dataset = create_dataset(preprocessor.test_keys)

# Concatenate batches
all_X_test = []
all_y_test = []
for X_batch, y_batch in test_dataset:
    all_X_test.append(X_batch.numpy())
    all_y_test.append(y_batch.numpy())
X_test = np.concatenate(all_X_test, axis=0)
y_test = np.concatenate(all_y_test, axis=0)

# Predict and compute metrics
y_pred = (trainer.predict(X_test) >= 0.5).astype(np.int8)
y_true = y_test.reshape(-1, 2)
y_pred = y_pred.reshape(-1, 2)

cm_knees = confusion_matrix(y_true[:, 0], y_pred[:, 0]).tolist()
cm_elbows = confusion_matrix(y_true[:, 1], y_pred[:, 1]).tolist()
accuracy = accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
precision_knees = precision_score(y_true[:, 0], y_pred[:, 0], zero_division=0)
precision_elbows = precision_score(y_true[:, 1], y_pred[:, 1], zero_division=0)
recall_knees = recall_score(y_true[:, 0], y_pred[:, 0], zero_division=0)
recall_elbows = recall_score(y_true[:, 1], y_pred[:, 1], zero_division=0)
f1_knees = f1_score(y_true[:, 0], y_pred[:, 0], zero_division=0)
f1_elbows = f1_score(y_true[:, 1], y_pred[:, 1], zero_division=0)

# Average metrics across knees and elbows
precision = (precision_knees + precision_elbows) / 2
recall = (recall_knees + recall_elbows) / 2
f1_score_avg = (f1_knees + f1_elbows) / 2

# Save metrics
metrics_json = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score_avg,
    "confusion_matrix_knees": cm_knees,
    "confusion_matrix_elbows": cm_elbows
}
metrics_path = os.path.join(metrics_dir, "lstm_metrics.json")
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
with open(metrics_path, 'w') as f:
    json.dump(metrics_json, f, indent=4)

print("Metrics saved to:", metrics_path)
print("Final Test Metrics:", metrics_json)