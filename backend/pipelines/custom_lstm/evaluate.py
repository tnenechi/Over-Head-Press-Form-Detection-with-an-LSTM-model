import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.preprocess import Preprocessor
from pipelines.custom_lstm.model import CustomLSTM

# Config 
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
data_dir = os.path.join(BASE_DIR, "data")
checkpoint_dir = os.path.join(BASE_DIR, "pipelines", "custom_lstm", "checkpoints")
metrics_dir = os.path.join(BASE_DIR, "pipelines", "custom_lstm", "metrics")
os.makedirs(metrics_dir, exist_ok=True)

batch_size = 16

# Preprocessor
preprocessor = Preprocessor(data_dir)
input_shape = (preprocessor.median_frames, 100)  # seq_length x features

# Load Custom LSTM model
model_path = os.path.join(checkpoint_dir, "custom_lstm_model.h5")
trainer = CustomLSTM(
    input_size=input_shape[1],
    hidden_sizes=[64, 128, 256, 128, 64],  # match training config
    output_size=2,
    seq_length=input_shape[0]
)
trainer._keras_model = tf.keras.models.load_model(model_path)

# Helper: load all test/val data
def load_all_data(keys):
    X, y = [], []
    for X_batch, y_batch in preprocessor.load_data(keys, batch_size=batch_size):
        X.append(X_batch)
        y.append(y_batch)
    if not X:
        return np.empty((0, input_shape[0], input_shape[1])), np.empty((0, input_shape[0], 2))
    X = np.vstack(X)
    y = np.vstack(y)
    return X, y

# Evaluate on train/val/test sets
metrics_dict = {}
for split_name, split_keys in [("train", preprocessor.train_keys),
                              ("val", preprocessor.val_keys),
                              ("test", preprocessor.test_keys)]:
    X, y = load_all_data(split_keys)
    if X.size == 0:
        print(f"No data found for {split_name} split, skipping.")
        continue
    raw_metrics = trainer.evaluate(X, y)  # returns per-joint confusion matrices
    metrics_dict[split_name] = {
        "confusion_matrix_knees": raw_metrics["confusion_matrix_knees"],
        "confusion_matrix_elbows": raw_metrics["confusion_matrix_elbows"],
        "accuracy": raw_metrics["accuracy"],
        "precision": raw_metrics["precision"],
        "recall": raw_metrics["recall"],
        "f1_score": raw_metrics["f1_score"]
    }

# Save metrics JSON in API-friendly format
metrics_path = os.path.join(metrics_dir, "custom_lstm_metrics.json")
if "test" in metrics_dict:
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict["test"], f, indent=4)  
    print(f"Metrics saved to {metrics_path}")