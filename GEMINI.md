# Project Overview

This project is a full-stack application designed for video analysis, focusing on human movement or activity. The system comprises a React-based frontend and a Flask-based backend, integrating various machine learning models for prediction and analysis.

## Architecture

*   **Frontend:** A React application built with Vite, styled using Tailwind CSS and DaisyUI. It provides a user interface for video uploads, model selection, displaying analysis results (predictions and keypoints), and visualizing model performance metrics and training history using Chart.js.
*   **Backend:** A Flask API responsible for video processing (frame extraction, keypoint detection using MediaPipe), running machine learning models, and serving model-related data (metrics, history). It utilizes TensorFlow/Keras, scikit-learn, OpenCV, and NumPy for its operations.

## Key Features

*   **Video Upload and Analysis:** Users can upload video files for analysis.
*   **Multiple ML Models:** The backend supports different machine learning pipelines, including LSTM, custom LSTM, and custom feedforward networks, for making predictions.
*   **Keypoint Detection:** Utilizes MediaPipe for detecting keypoints in video frames.
*   **Performance Metrics:** Provides endpoints to retrieve and display performance metrics and training history for the integrated ML models.
*   **Data Statistics:** Offers insights into the distribution of error types within the datasets.

# Building and Running

## Frontend

The frontend is a React application.

### Dependencies

Install dependencies using npm:
```bash
npm install
```

### Development Server

To run the development server:
```bash
npm run dev
```
The application will typically be available at `http://localhost:5173` (or another port as indicated by Vite).

### Build

To build the production-ready application:
```bash
npm run build
```
The build artifacts will be located in the `dist` directory.

### Linting

To lint the frontend code:
```bash
npm run lint
```

## Backend

The backend is a Flask application.

### Dependencies

Install Python dependencies from `requirements.txt`:
```bash
pip install -r backend/requirements.txt
```

### Running the API

To run the Flask development server:
```bash
python backend/api.py
```
The API will run on `http://0.0.0.0:5000`.

### API Endpoints

*   **`POST /predict`**: Upload a video for analysis and get predictions.
    *   Parameters: `video` (file), `model` (string, e.g., 'lstm', 'custom_lstm', 'custom_feedforward')
*   **`GET /metrics/<pipeline>`**: Get metrics for a specific pipeline.
    *   Parameter: `pipeline` (string, e.g., 'lstm', 'custom_lstm', 'custom_feedforward')
*   **`GET /metrics/combined`**: Get metrics for all pipelines.
*   **`GET /history/<pipeline>`**: Get training history for a specific pipeline.
    *   Parameter: `pipeline` (string, e.g., 'lstm', 'custom_lstm', 'custom_feedforward')
*   **`GET /data_stats`**: Get statistics about the datasets.

# Development Conventions

## Code Structure

*   **Frontend:** Standard React project structure with `src/pages` for main views, `src/components` for reusable UI elements.
*   **Backend:** Flask application with `pipelines` directory containing different ML model implementations (LSTM, custom LSTM, custom feedforward), and `utils` for shared functionalities like preprocessing.

## Styling

The frontend uses Tailwind CSS and DaisyUI for styling.

## Data

*   The `backend/data` directory contains `labelled_dataset` (with labels, splits, and video files) and `unlabelled_dataset`.
*   Model checkpoints and metrics are stored within their respective pipeline directories (e.g., `backend/pipelines/lstm/checkpoints`, `backend/pipelines/lstm/metrics`).
