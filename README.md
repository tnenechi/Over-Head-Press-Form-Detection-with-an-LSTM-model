# TechniqueAI

AI-powered overhead press form analysis. Detects incorrect elbow and knee alignment in real-time to improve lifting technique and prevent injuries.

Advanced AI models analyze video input frame-by-frame, focusing on elbow flare, knee cave, and other common overhead press mistakes.

[Read the full technical report (PDF)](docs/paper.pdf)

## Features

- Real-time detection of improper elbow and knee positioning during overhead presses
- Multiple model architectures: LSTM, Custom LSTM, Feedforward
- Video upload & keypoint extraction (MediaPipe)
- Frame-level predictions for knees and elbows
- REST API endpoint for integration with frontends/mobile apps
- Training pipelines with checkpointing and metrics tracking
- Evaluation metrics (accuracy, precision, recall, F1) per joint

## Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow/Keras
- **Computer Vision**: MediaPipe, OpenCV
- **Data Processing**: NumPy, scikit-learn
- **API**: Flask + CORS
- **Frontend**: React, ChartJS, Tailwind
