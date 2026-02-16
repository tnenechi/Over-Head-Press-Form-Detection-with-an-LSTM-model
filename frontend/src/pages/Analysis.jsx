import { useState, useRef, useEffect } from "react";
import VideoPlayer from "../components/VideoPlayer";
import ErrorDisplay from "../components/ErrorDisplay";
import { Link } from "react-router-dom";

function Analysis() {
  const [videoFile, setVideoFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [keypoints, setKeypoints] = useState(null);
  const [selectedModel, setSelectedModel] = useState("custom_lstm");
  const [metrics, setMetrics] = useState({});
  const [isProcessing, setIsProcessing] = useState(false);
  const videoRef = useRef(null);

  const models = [
    {
      id: "custom_lstm",
      name: "Custom LSTM",
      description:
        "An LSTM model capturing temporal dependencies in video sequences.",
    },
    {
      id: "lstm",
      name: "Inception LSTM",
      description: "An enhanced LSTM model for video-based pose correction.",
    },
    {
      id: "custom_feedforward",
      name: "FeedForward",
      description:
        "A feedforward neural network for frame-by-frame pose analysis.",
    },
  ];

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch(`http://localhost:5000/metrics/combined`);
        if (!response.ok) throw new Error("Failed to fetch metrics");
        const data = await response.json();
        setMetrics(data);
      } catch (error) {
        console.error("Error fetching metrics:", error);
        setMetrics({});
      }
    };
    fetchMetrics();
    setVideoFile(null); // Clear video file and related data on model change
    setPredictions(null);
    setKeypoints(null);
  }, [selectedModel]);

  const handleVideoUpload = (file) => {
    setVideoFile(null); // Clear video file and related data on new file selection
    setPredictions(null);
    setKeypoints(null);
    setIsProcessing(true);
    setTimeout(() => setVideoFile(file), 0); // Defer setting new file to allow UI update
  };

  const handlePredictions = (data) => {
    setPredictions(data.predictions);
    setKeypoints(data.keypoints);
    setIsProcessing(false);
  };

  return (
    <div className="container mx-auto p-4 bg-gray-100 text-gray-500 min-h-screen flex flex-col">
      <div className="shadow-md p-3 mb-4">
        <Link
          to="/"
          className="text-info hover:underline underline-offset-2 transition"
        >
          Home
        </Link>
      </div>

      <h1 className="text-3xl font-bold mb-4 text-center">
        Video Analysis
      </h1>

      <div className="mb-15 mt-15 relative bg-white p-4 rounded-lg shadow-md max-w-1/4">
        {/* <h2 className="absolute -top-11 ring-1 font-semibold  left-0 bg-gray-300 rounded-full h-10 w-10 flex items-center justify-center">
          1
        </h2> */}
        <h2 className="text-xl font-semibold mb-2">Select Analysis Model</h2>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full p-2 border rounded-md focus:ring-2 focus:ring-blue-500"
        >
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>
        <p className="mt-2 text-gray-600">
          {models.find((m) => m.id === selectedModel)?.description}
        </p>
      </div>

      <div className="flex flex-col md:flex-row gap-4 mt-5">
        <div className="mb-4 relative bg-white p-4 rounded-lg shadow-md md:w-1/2">
          {/* <h2 className="shadow-lg absolute -top-11 ring-1 font-semibold  left-0 bg-gradient-to-t  from-green-600 from-60% to-gray-300 to-40% rounded-full h-10 w-10 flex items-center justify-center">
            2
          </h2> */}
          <h2 className="text-xl font-semibold mb-2">Upload Your Video</h2>
          <p className="text-gray-600 mb-4 ">
            Please upload a video (MP4 format) to be analyzed for pose errors.
            Ensure the video clearly shows body movements.
          </p>
          <VideoPlayer
            onVideoUpload={handleVideoUpload}
            onPredictions={handlePredictions}
            selectedModel={selectedModel}
            predictions={predictions}
            keypoints={keypoints}
            videoRef={videoRef}
          />
        </div>

        <div className="relative">
          {/* <h2 className="shadow-lg absolute -top-11 ring-2 font-semibold  left-0 bg-green-600 rounded-full h-10 w-10 flex items-center justify-center">
            3
          </h2> */}
          {videoFile &&
            !isProcessing &&
            predictions &&
            Array.isArray(predictions) &&
            predictions.every((p) => Array.isArray(p) && p.length === 2) && (
              <div className="bg-blue p-4 rounded-lg shadow-md">
                <ErrorDisplay
                  predictions={predictions}
                  modelName={models.find((m) => m.id === selectedModel)?.name}
                  videoRef={videoRef}
                  metrics={metrics[selectedModel] || {}}
                />
              </div>
            )}

          {videoFile &&
            !isProcessing &&
            (!predictions ||
              !Array.isArray(predictions) ||
              !predictions.every(
                (p) => Array.isArray(p) && p.length === 2,
              )) && (
              <div className="text-center text-red-500 w-full md:w-1/2">
                Invalid prediction data received.
              </div>
            )}
        </div>
      </div>
    </div>
  );
}

export default Analysis;
