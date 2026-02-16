import React, { useState } from "react";

function ErrorDisplay({ predictions, modelName, videoRef, metrics }) {
  const [showModal, setShowModal] = useState(false);

  console.log("Predictions in ErrorDisplay:", predictions);

  if (!predictions || !Array.isArray(predictions) || predictions.length === 0) {
    return (
      <div className="mt-6 bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-xl font-bold mb-4 text-gray-800">
          {modelName} Analysis Results
        </h3>
        <p className="text-red-500 bg-red-100 p-3 rounded-lg">
          No valid predictions available for analysis.
        </p>
      </div>
    );
  }

  const frameRate = 30;
  const intervalSeconds = 5;
  const framesPerInterval = intervalSeconds * frameRate;

  const errorIntervals = [];
  for (let i = 0; i < predictions.length; i += framesPerInterval) {
    const intervalFrames = predictions.slice(i, i + framesPerInterval);
    const hasError =
      intervalFrames.every(
        (frame) => Array.isArray(frame) && frame.length === 2
      ) && intervalFrames.some((frame) => frame[0] >= 0.5 || frame[1] >= 0.5);
    if (hasError) {
      errorIntervals.push({
        time: i / frameRate,
        interval: Math.floor(i / framesPerInterval) + 1,
      });
    }
  }

  const totalFrames = predictions.length;
  const errorFrames = predictions.filter(
    (frame) =>
      Array.isArray(frame) &&
      frame.length === 2 &&
      (frame[0] >= 0.5 || frame[1] >= 0.5)
  ).length;
  const performanceScore = (
    ((totalFrames - errorFrames) / totalFrames) *
    100
  ).toFixed(1);

  const jumpToError = (time) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      videoRef.current.play();
    }
  };

  const confusionMatrix = metrics.confusion_matrix || {
    knees: [
      [0, 0],
      [0, 0],
    ],
    elbows: [
      [0, 0],
      [0, 0],
    ],
  };

  console.log("Confusion Matrix:", confusionMatrix);

  const kneeError = predictions.some((frame) => frame[0] >= 0.5);
  const elbowError = predictions.some((frame) => frame[1] >= 0.5);
  const errorType =
    kneeError && elbowError
      ? "Knee and Elbow Error Detected"
      : kneeError
      ? "Knee Error Detected"
      : elbowError
      ? "Elbow Error Detected"
      : "No Errors Detected";

  return (
    <div className="mt-6 bg-white p-6 rounded-lg shadow-md">
      <h3 className="text-xl font-bold mb-4 text-gray-800">
        {modelName} Analysis Results
      </h3>

      <div className="mb-6">
        <div className="flex items-center space-x-2">
          <h4 className="text-lg font-semibold">
            Performance Score:{" "}
            <span className="text-blue-600">{performanceScore}%</span>
          </h4>
          <button
            onClick={() => setShowModal(true)}
            className="text-gray-500 hover:text-gray-700"
            title="Score Info"
          >
            <svg
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </button>
        </div>
      </div>

      <p
        className={`${
          errorType !== "No Errors Detected"
            ? "text-red-600 bg-red-100"
            : "text-green-600 bg-green-100"
        } p-3 rounded`}
      >
        {errorType}
      </p>

      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg max-w-md w-full">
            <h4 className="text-lg font-semibold mb-2">Performance Score</h4>
            <p className="text-gray-700 mb-4">
              The performance score represents the percentage of video frames
              where no pose errors (knees or elbows) were detected by the{" "}
              {modelName} model. A higher score indicates better form throughout
              the video.
            </p>
            <button
              onClick={() => setShowModal(false)}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default ErrorDisplay;
