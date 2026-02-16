# TechniqueAI Project Overview

TechniqueAI is a full-stack application designed to provide AI-powered analysis of overhead press form. It detects incorrect elbow and knee alignment in real-time from video input to help users improve their lifting technique and prevent injuries. The project integrates advanced machine learning models with a user-friendly web interface.

## Architecture

The project consists of two main components: a Python-based backend and a React-based frontend.

### Backend

*   **Technology Stack**: Python, Flask, TensorFlow/Keras, MediaPipe, OpenCV, NumPy, scikit-learn.
*   **Functionality**:
    *   Provides a REST API for video analysis, model evaluation metrics, training history, and data statistics.
    *   Utilizes MediaPipe for keypoint extraction from video frames.
    *   Employs various machine learning models (LSTM, Custom LSTM, Feedforward) for form analysis.
    *   Manages model training, checkpointing, and metric tracking.
*   **Key Directories**:
    *   `backend/api.py`: The main Flask application defining API endpoints.
    *   `backend/pipelines/`: Contains the implementation for different machine learning models and their associated training, evaluation, and model files.
    *   `backend/utils/preprocess.py`: Handles video frame extraction, keypoint detection, and data preprocessing.

### Frontend

*   **Technology Stack**: React, Vite, ChartJS, Tailwind CSS, Axios, React Router DOM.
*   **Functionality**:
    *   A single-page application that interacts with the backend API.
    *   Allows users to upload videos for analysis.
    *   Displays model evaluation dashboards with training history, performance metrics, and confusion matrices using ChartJS.
    *   Provides navigation and user interface components.
*   **Key Directories**:
    *   `frontend/src/main.jsx`, `frontend/src/App.jsx`: Core React application entry points.
    *   `frontend/src/components/`: Reusable React components (e.g., `VideoPlayer.jsx`, `ErrorDisplay.jsx`).
    *   `frontend/src/pages/`: Page-specific React components (e.g., `Home.jsx`, `Analysis.jsx`, `Dashboard.jsx`).

## Building and Running

### Backend Setup

1.  **Navigate to the backend directory**:
    ```bash
    cd backend
    ```
2.  **Create and activate a Python virtual environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate # On Windows
    # source venv/bin/activate # On macOS/Linux
    ```
3.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Flask API server**:
    ```bash
    python api.py
    ```
    The API will typically run on `http://localhost:5000`.

### Frontend Setup

1.  **Navigate to the frontend directory**:
    ```bash
    cd frontend
    ```
2.  **Install Node.js dependencies**:
    ```bash
    npm install
    ```
3.  **Run the React development server**:
    ```bash
    npm run dev
    ```
    The frontend application will typically run on `http://localhost:5173` (or another port specified by Vite).

## API Endpoints

The Flask backend exposes the following key API endpoints:

*   **`/predict` (POST)**: Accepts a video file and a `model` name. Returns frame-level predictions (knees and elbows risk) and detected keypoints.
*   **`/metrics/<pipeline>` (GET)**: Returns evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix) for a specific machine learning pipeline (`lstm`, `custom_lstm`, or `custom_feedforward`).
*   **`/metrics/combined` (GET)**: Returns evaluation metrics for all available machine learning pipelines.
*   **`/history/<pipeline>` (GET)**: Returns the training history (accuracy, loss, validation accuracy, validation loss per epoch) for a specific pipeline.
*   **`/data_stats` (GET)**: Provides statistics about the distribution of errors (knees, elbows, none) within the training, validation, and test datasets.

## Machine Learning Pipelines

The project includes three distinct machine learning pipelines, each designed for different architectural approaches to sequence modeling:

1.  **Inception LSTM (`lstm`)**: Utilizes an Inception-like module with parallel LSTM layers and residual connections to process video sequences and make frame-level predictions.
2.  **Custom LSTM (`custom_lstm`)**: A custom-built LSTM architecture tailored for tracking movements and making frame-level risk predictions.
3.  **FeedForward (`custom_feedforward`)**: A simpler architecture that analyzes each frame separately using dense layers to provide risk predictions.

## Development Conventions

*   **Frontend Linting**: ESLint is configured for the frontend (`eslint.config.js`) to maintain code quality and consistency.
*   **Configuration**: Frontend build process is managed by Vite (`vite.config.js`).
*   **Styling**: Tailwind CSS is used for styling the frontend.
