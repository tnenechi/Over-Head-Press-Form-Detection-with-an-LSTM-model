import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { LifeLine } from "react-loading-indicators";

function VideoPlayer({
  onVideoUpload,
  onPredictions,
  selectedModel,
  predictions,
  keypoints,
  videoRef,
}) {
  const [videoUrl, setVideoUrl] = useState("");
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const canvasRef = useRef(null);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    setError("");
    setVideoUrl(""); // Clear video playback immediately
    onVideoUpload(file);

    const formData = new FormData();
    formData.append("video", file);
    formData.append("model", selectedModel);

    try {
      const response = await axios.post(
        "http://localhost:5000/predict",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      console.log("Backend response:", response.data);
      setVideoUrl(URL.createObjectURL(file));
      onPredictions(response.data);
    } catch (err) {
      setError(
        "Error uploading video: " + (err.response?.data?.error || err.message)
      );
    } finally {
      setUploading(false);
    }
  };

  useEffect(() => {
    if (
      !videoRef.current ||
      !canvasRef.current ||
      !predictions ||
      !keypoints ||
      !predictions.length ||
      !keypoints.length
    ) {
      console.log(
        "Skipping drawFrame: missing video, canvas, predictions, or keypoints"
      );
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const frameRate = 30;
    const frameCount = Math.min(predictions.length, keypoints.length);

    const resizeCanvas = () => {
      canvas.width = video.offsetWidth;
      canvas.height = video.offsetHeight;
    };

    const drawFrame = () => {
      if (video.paused || video.ended) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }

      const currentFrame = Math.floor(video.currentTime * frameRate);
      if (currentFrame >= frameCount) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }

      const framePredictions = predictions[currentFrame];
      const frameKeypoints = keypoints[currentFrame];

      if (!framePredictions || !frameKeypoints) {
        console.log(
          `Invalid data at frame ${currentFrame}: predictions=${framePredictions}, keypoints=${frameKeypoints}`
        );
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const videoWidth = video.videoWidth || 224;
      const videoHeight = video.videoHeight || 224;
      const displayWidth = video.offsetWidth;
      const displayHeight = video.offsetHeight;
      const scaleX = displayWidth / videoWidth;
      const scaleY = displayHeight / videoHeight;
      const scale = Math.min(scaleX, scaleY);
      const offsetX = (displayWidth - videoWidth * scale) / 2;
      const offsetY = (displayHeight - videoHeight * scale) / 2;

      if (framePredictions[0] >= 0.5 || framePredictions[1] >= 0.5) {
        const landmarksToDraw = [
          { index: 12, error: framePredictions[1] >= 0.5, label: "elbow" },
          { index: 13, error: framePredictions[1] >= 0.5, label: "elbow" },
          { index: 24, error: framePredictions[0] >= 0.5, label: "knee" },
          { index: 25, error: framePredictions[0] >= 0.5, label: "knee" },
        ];

        landmarksToDraw.forEach(({ index, error }) => {
          if (
            error &&
            frameKeypoints[index] &&
            frameKeypoints[index][0] &&
            frameKeypoints[index][1]
          ) {
            const x = frameKeypoints[index][0] * scale + offsetX;
            const y = frameKeypoints[index][1] * scale + offsetY;
            console.log(
              `Drawing at frame ${currentFrame}, landmark ${index}: x=${x}, y=${y}`
            );
            ctx.beginPath();
            ctx.arc(x, y, 10, 0, 2 * Math.PI);
            ctx.fillStyle = "red";
            ctx.globalAlpha = 0.6;
            ctx.fill();
            ctx.globalAlpha = 1.0;
          }
        });
      }

      requestAnimationFrame(drawFrame);
    };

    video.addEventListener("loadedmetadata", resizeCanvas);
    window.addEventListener("resize", resizeCanvas);
    video.addEventListener("play", () => {
      resizeCanvas();
      drawFrame();
    });

    return () => {
      video.removeEventListener("loadedmetadata", resizeCanvas);
      window.removeEventListener("resize", resizeCanvas);
      video.removeEventListener("play", drawFrame);
    };
  }, [predictions, keypoints, videoRef]);

  return (
    <div className="mb-4 relative">
      <label
        htmlFor="video-upload"
        className="block text-sm font-medium text-gray-700 mb-2"
      >
        Choose a video file
      </label>
      <input
        id="video-upload"
        type="file"
        accept="video/mp4"
        onChange={handleFileChange}
        className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-blue-500 file:text-white hover:file:bg-blue-600 disabled:opacity-50"
        disabled={uploading}
      />
      {uploading && (
        <div className="mt-2 flex items-center justify-center space-x-2">
          <LifeLine color="#3b82f6" size="small" text="" textColor="" />
          <p className="text-blue-500 text-xs">Uploading and analyzing...</p>
        </div>
      )}
      {videoUrl && (
        <div className="relative mt-4 max-w-xl mx-auto">
          <video
            ref={videoRef}
            controls
            src={videoUrl}
            className="rounded-lg shadow-md w-full"
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 pointer-events-none w-full h-full"
          />
        </div>
      )}
      {error && (
        <p className="mt-4 text-red-500 bg-red-100 p-3 rounded-lg">{error}</p>
      )}
    </div>
  );
}

export default VideoPlayer;
