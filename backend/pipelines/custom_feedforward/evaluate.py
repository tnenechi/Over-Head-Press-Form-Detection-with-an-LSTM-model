import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow oneDNN logs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.preprocess import Preprocessor
from pipelines.custom_feedforward.model import FeedforwardTrainer
import numpy as np
import json
from sklearn.metrics import confusion_matrix

preprocessor = Preprocessor("./data")
input_shape = (preprocessor.median_frames, 100)

model_path = "./pipelines/custom_feedforward/checkpoints/custom_feedforward_model.npz"
trainer = FeedforwardTrainer(input_shape)
trainer.load_weights(model_path)

def load_all_data(keys):
    X, y = [], []
    for X_batch, y_batch in preprocessor.load_data(keys, batch_size=32):
        X.append(X_batch)
        y.append(y_batch)
    X = np.vstack(X)
    y = np.vstack(y)
    print(f"Test data shape: X={X.shape}, y={y.shape}")
    return X, y

X_test, y_test = load_all_data(preprocessor.test_keys)

# Verify input shape
if X_test.shape[2] != input_shape[1]:
    raise ValueError(f"Input feature mismatch: Expected {input_shape[1]} features, got {X_test.shape[2]}")

metrics = trainer.evaluate(X_test, y_test)

# Compute separate confusion matrices for knees and elbows
y_pred = (trainer.predict(X_test) >= 0.5).astype(np.int8)
y_true = y_test.reshape(-1, 2)
y_pred = y_pred.reshape(-1, 2)
cm_knees = confusion_matrix(y_true[:, 0], y_pred[:, 0]).tolist()
cm_elbows = confusion_matrix(y_true[:, 1], y_pred[:, 1]).tolist()

metrics_json = {
    "accuracy": metrics['accuracy'],
    "precision": metrics['precision'],
    "recall": metrics['recall'],
    "f1_score": metrics['f1_score'],
    "confusion_matrix": {
        "knees": cm_knees,
        "elbows": cm_elbows
    }
}
metrics_path = "./pipelines/custom_feedforward/metrics/custom_feedforward_metrics.json"
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
with open(metrics_path, 'w') as f:
    json.dump(metrics_json, f, indent=4)

print("Metrics saved to:", metrics_path)
print("Final Test Metrics:", metrics_json)

