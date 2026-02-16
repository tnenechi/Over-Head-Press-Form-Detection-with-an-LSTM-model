from flask import Flask, request, jsonify, send_from_directory
from utils.preprocess import Preprocessor
from pipelines.lstm.model import LSTMTrainer
from pipelines.custom_lstm.model import CustomLSTM
from pipelines.custom_feedforward.model import FeedforwardTrainer
import numpy as np
from tensorflow.keras.models import load_model
import os
import json
from flask_cors import CORS

FRONTEND_BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'dist')
app = Flask(__name__, static_folder=FRONTEND_BUILD_DIR, static_url_path='/')
CORS(app)

data_dir = "./data"
lstm_model_path = "./pipelines/lstm/checkpoints/lstm_model.h5"
custom_lstm_model_path = "./pipelines/custom_lstm/checkpoints/custom_lstm_model.h5"
custom_feedforward_model_path = "./pipelines/custom_feedforward/checkpoints/custom_feedforward_model.npz"

preprocessor = Preprocessor(data_dir)
input_shape = (113, 100)  # Hardcode to match trained model

try:
    lstm_model = load_model(lstm_model_path)
    lstm_trainer = LSTMTrainer(input_shape)
    lstm_trainer.model = lstm_model
except Exception as e:
    lstm_trainer = None
    print(f"Error loading LSTM model: {e}")

try:
    custom_lstm_model = load_model(custom_lstm_model_path)
    custom_lstm_trainer = CustomLSTM(
        input_size=100,
        hidden_sizes=[64, 128, 256, 128, 64],
        output_size=2,
        seq_length=113  # Hardcode to match trained model
    )
    custom_lstm_trainer._keras_model = custom_lstm_model  # Assign the loaded model
except Exception as e:
    custom_lstm_trainer = None
    print(f"Error loading Custom LSTM model: {e}")

try:
    custom_feedforward_trainer = FeedforwardTrainer(input_shape)
    custom_feedforward_trainer.load_weights(custom_feedforward_model_path)
except Exception as e:
    custom_feedforward_trainer = None
    print(f"Error loading Custom Feedforward model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    model_name = request.form.get('model', 'custom_feedforward')
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if model_name not in ['lstm', 'custom_lstm', 'custom_feedforward']:
        return jsonify({"error": "Invalid model name"}), 400
    
    temp_video_path = os.path.join(data_dir, "unlabelled_dataset", f"temp_video_{model_name}.mp4")
    os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
    video_file.save(temp_video_path)
    
    frames, _ = preprocessor.extract_frames(temp_video_path)
    if not frames:
        os.remove(temp_video_path)
        return jsonify({"error": "No frames extracted from video"}), 400
    keypoints = preprocessor.detect_keypoints(frames)
    if not keypoints.size:
        os.remove(temp_video_path)
        return jsonify({"error": "No keypoints detected"}), 400
    keypoints = preprocessor.interpolate_keypoints(keypoints, 113)  # Explicitly use 113
    diff = np.diff(keypoints, axis=0, prepend=keypoints[0:1])
    # Pad or truncate to 113 frames
    if diff.shape[0] < 113:
        diff = np.pad(diff, ((0, 113 - diff.shape[0]), (0, 0)), mode='constant')
    elif diff.shape[0] > 113:
        diff = diff[:113]
    X = diff.reshape(1, 113, 100)  # Ensure batch dimension and correct shape
    
    trainer = {
        'lstm': lstm_trainer,
        'custom_lstm': custom_lstm_trainer,
        'custom_feedforward': custom_feedforward_trainer,
    }[model_name]
    
    if not trainer:
        os.remove(temp_video_path)
        return jsonify({"error": f"{model_name} model not loaded"}), 400
    
    predictions = trainer.predict(X)[0]  # Shape: (113, 2)
    print(f"Predictions for {model_name}: {predictions}")
    keypoints_normalized = keypoints[:, :25 * 4].reshape(-1, 25, 4)
    keypoints_xy = keypoints_normalized[:, :, :2] * [224, 224]
    print(f"Backend: model={model_name}, predictions_shape={predictions.shape}, keypoints_shape={keypoints_xy.shape}")
    
    os.remove(temp_video_path)
    
    return jsonify({
        'predictions': predictions.tolist(),
        'keypoints': keypoints_xy.tolist()
    })

@app.route('/metrics/<pipeline>', methods=['GET'])
def get_metrics(pipeline):
    metrics_paths = {
        'lstm': "./pipelines/lstm/metrics/lstm_metrics.json",
        'custom_lstm': "./pipelines/custom_lstm/metrics/custom_lstm_metrics.json",
        'custom_feedforward': "./pipelines/custom_feedforward/metrics/custom_feedforward_metrics.json",
    }
    if pipeline not in metrics_paths:
        return jsonify({"error": "Invalid pipeline name"}), 400
    metrics_path = metrics_paths[pipeline]
    if not os.path.exists(metrics_path):
        return jsonify({"error": f"{pipeline.capitalize()} metrics not found."}), 404
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return jsonify(metrics)

@app.route('/metrics/combined', methods=['GET'])
def get_combined_metrics():
    metrics = {}
    for pipeline in ['lstm', 'custom_lstm', 'custom_feedforward']:
        metrics_paths = {
            'lstm': "./pipelines/lstm/metrics/lstm_metrics.json",
            'custom_lstm': "./pipelines/custom_lstm/metrics/custom_lstm_metrics.json",
            'custom_feedforward': "./pipelines/custom_feedforward/metrics/custom_feedforward_metrics.json",
        }
        metrics_path = metrics_paths[pipeline]
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics[pipeline] = json.load(f)
        else:
            metrics[pipeline] = {"error": f"{pipeline.capitalize()} metrics not found"}
    return jsonify(metrics)

@app.route('/history/<pipeline>', methods=['GET'])
def get_history(pipeline):
    history_dirs = {
        'lstm': "./pipelines/lstm/checkpoints",
        'custom_lstm': "./pipelines/custom_lstm/checkpoints",
        'custom_feedforward': "./pipelines/custom_feedforward/checkpoints",
    }
    if pipeline not in history_dirs:
        return jsonify({"error": "Invalid pipeline name"}), 400
    history_dir = history_dirs[pipeline]
    history_dict = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    for epoch in range(1, 51):
        history_file = os.path.join(history_dir, f"history_epoch_{epoch}.npy")
        if os.path.exists(history_file):
            h = np.load(history_file, allow_pickle=True).item()
            for key in history_dict:
                if key in h:
                    history_dict[key].append(h[key][0] if isinstance(h[key], list) else h[key])
    return jsonify(history_dict)

@app.route('/data_stats', methods=['GET'])
def get_data_stats():
    train_error_counts = {'knees': 0, 'elbows': 0, 'none': 0}
    val_error_counts = {'knees': 0, 'elbows': 0, 'none': 0}
    test_error_counts = {'knees': 0, 'elbows': 0, 'none': 0}
    
    def count_errors(keys, error_counts):
        for key in keys:
            knees = len(preprocessor.error_knees.get(key, [])) > 0
            elbows = len(preprocessor.error_elbows.get(key, [])) > 0
            if knees and elbows:
                error_counts['knees'] += 1
                error_counts['elbows'] += 1
            elif knees:
                error_counts['knees'] += 1
            elif elbows:
                error_counts['elbows'] += 1
            else:
                error_counts['none'] += 1
    
    count_errors(preprocessor.train_keys, train_error_counts)
    count_errors(preprocessor.val_keys, val_error_counts)
    count_errors(preprocessor.test_keys, test_error_counts)
    
    return jsonify({
        'train': train_error_counts,
        'val': val_error_counts,
        'test': test_error_counts
    })

    
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
