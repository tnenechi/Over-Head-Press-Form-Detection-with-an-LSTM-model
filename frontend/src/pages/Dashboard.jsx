import React, { useState, useEffect } from "react";
import { Bar, Line, Pie } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Filler,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Link } from "react-router-dom";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Filler,
  Title,
  Tooltip,
  Legend,
);

function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState(null);
  const [dataStats, setDataStats] = useState(null);
  const [selectedPipeline, setSelectedPipeline] = useState("custom_lstm");
  const [error, setError] = useState("");

  const pipelines = ["custom_lstm", "lstm", "custom_feedforward"];
  const pipelineDisplayNames = {
    custom_lstm: "Custom LSTM",
    lstm: "Inception LSTM",
    custom_feedforward: "FeedForward",
  };

  useEffect(() => {
    async function fetchData() {
      try {
        const metricsResponse = await fetch("/metrics/combined");
        if (!metricsResponse.ok) throw new Error("Failed to fetch metrics");
        const metricsData = await metricsResponse.json();
        setMetrics(metricsData);
        console.log("Metrics data from backend:", metricsData);

        const historyPromises = pipelines.map(async (pipeline) => {
          const response = await fetch(`/history/${pipeline}`);
          if (!response.ok)
            return {
              [pipeline]: {
                error: `Failed to fetch ${pipelineDisplayNames[pipeline]} history`,
              },
            };
          return { [pipeline]: await response.json() };
        });
        const historyResults = await Promise.all(historyPromises);
        const historyData = Object.assign({}, ...historyResults);
        setHistory(historyData);

        const statsResponse = await fetch("/data_stats");
        if (!statsResponse.ok) throw new Error("Failed to fetch data stats");
        const statsData = await statsResponse.json();
        setDataStats(statsData);
      } catch (err) {
        setError("Error loading data: " + err.message);
      }
    }
    fetchData();
  }, []);

  if (error)
    return <div className="text-red-500 text-center text-lg">{error}</div>;
  if (!metrics || !history || !dataStats)
    return (
      <div className="text-gray-500 text-center text-lg">
        Loading dashboard...
      </div>
    );

  const historyChartData = {
    labels: Array.from(
      {
        length: Math.max(
          ...pipelines.map((p) => history[p]?.accuracy?.length || 0),
        ),
      },
      (_, i) => i + 1,
    ),
    datasets: [
      {
        label: "Train Accuracy",
        data: history[selectedPipeline]?.accuracy || [],
        borderColor: "rgba(75, 192, 192, 1)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        fill: true,
      },
      {
        label: "Val Accuracy",
        data: history[selectedPipeline]?.val_accuracy || [],
        borderColor: "rgba(255, 99, 132, 1)",
        backgroundColor: "rgba(255, 99, 132, 0.2)",
        fill: true,
      },
      ...(selectedPipeline === "lstm"
        ? [
            {
              label: "Train Loss",
              data: history[selectedPipeline]?.loss || [],
              borderColor: "rgba(54, 162, 235, 1)",
              backgroundColor: "rgba(54, 162, 235, 0.2)",
              fill: true,
            },
            {
              label: "Val Loss",
              data: history[selectedPipeline]?.val_loss || [],
              borderColor: "rgba(255, 206, 86, 1)",
              backgroundColor: "rgba(255, 206, 86, 0.2)",
              fill: true,
            },
          ]
        : []),
    ],
  };

  const historyChartOptions = {
    scales: {
      x: { title: { display: true, text: "Epoch" } },
      y: { beginAtZero: true, max: 1, title: { display: true, text: "Value" } },
    },
    plugins: {
      title: {
        display: true,
        text: `${pipelineDisplayNames[selectedPipeline]} Training History`,
      },
    },
    maintainAspectRatio: false,
    responsive: true,
  };

  const metricChartData = {
    labels: ["Accuracy", "Precision", "Recall", "F1-Score"],
    datasets: [
      {
        label: pipelineDisplayNames[selectedPipeline],
        data: metrics[selectedPipeline]?.error
          ? [0, 0, 0, 0]
          : [
              metrics[selectedPipeline].accuracy,
              metrics[selectedPipeline].precision,
              metrics[selectedPipeline].recall,
              metrics[selectedPipeline].f1_score,
            ],
        backgroundColor: "rgba(75, 192, 192, 0.4)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
      },
    ],
  };

  const metricChartOptions = {
    scales: { y: { beginAtZero: true, max: 1 } },
    plugins: {
      title: {
        display: true,
        text: `${pipelineDisplayNames[selectedPipeline]} Performance Metrics`,
      },
    },
    maintainAspectRatio: false,
    responsive: true,
  };

  const confusionMatrixData = {
    datasets: [
      {
        label: `${pipelineDisplayNames[selectedPipeline]} Confusion Matrix (Knees)`,
        data:
          metrics[selectedPipeline]?.error ||
          !(
            (metrics[selectedPipeline].confusion_matrix?.knees &&
              Array.isArray(
                metrics[selectedPipeline].confusion_matrix.knees,
              )) ||
            (metrics[selectedPipeline].confusion_matrix_knees &&
              Array.isArray(metrics[selectedPipeline].confusion_matrix_knees))
          )
            ? []
            : (
                metrics[selectedPipeline].confusion_matrix?.knees ||
                metrics[selectedPipeline].confusion_matrix_knees
              ).flatMap((row, i) =>
                row.map((value, j) => ({
                  x: j,
                  y: i,
                  v: value,
                })),
              ),
        backgroundColor: (ctx) => {
          const value = ctx.raw?.v || 0;
          return value > 0
            ? ctx.raw.x === ctx.raw.y
              ? "rgba(0, 255, 0, 0.5)"
              : "rgba(255, 0, 0, 0.5)"
            : "rgba(0, 0, 0, 0.1)";
        },
        borderColor: "rgba(0, 0, 0, 1)",
        borderWidth: 1,
      },
      {
        label: `${pipelineDisplayNames[selectedPipeline]} Confusion Matrix (Elbows)`,
        data:
          metrics[selectedPipeline]?.error ||
          !(
            (metrics[selectedPipeline].confusion_matrix?.elbows &&
              Array.isArray(
                metrics[selectedPipeline].confusion_matrix.elbows,
              )) ||
            (metrics[selectedPipeline].confusion_matrix_elbows &&
              Array.isArray(metrics[selectedPipeline].confusion_matrix_elbows))
          )
            ? []
            : (
                metrics[selectedPipeline].confusion_matrix?.elbows ||
                metrics[selectedPipeline].confusion_matrix_elbows
              ).flatMap((row, i) =>
                row.map((value, j) => ({
                  x: j,
                  y: i,
                  v: value,
                })),
              ),
        backgroundColor: (ctx) => {
          const value = ctx.raw?.v || 0;
          return value > 0
            ? ctx.raw.x === ctx.raw.y
              ? "rgba(0, 255, 0, 0.5)"
              : "rgba(255, 0, 0, 0.5)"
            : "rgba(0, 0, 0, 0.1)";
        },
        borderColor: "rgba(0, 0, 0, 1)",
        borderWidth: 1,
      },
    ],
  };

  const confusionMatrixOptions = {
    scales: {
      x: {
        ticks: { callback: (v) => ["No Risk", "Risk"][v] },
        title: { display: true, text: "Predicted" },
      },
      y: {
        ticks: { callback: (v) => ["No Risk", "Risk"][v] },
        title: { display: true, text: "Actual" },
        reverse: true,
      },
    },
    plugins: {
      title: {
        display: true,
        text: `${pipelineDisplayNames[selectedPipeline]} Confusion Matrix`,
      },
    },
    maintainAspectRatio: false,
    responsive: true,
  };

  const comparisonBarData = {
    labels: ["Accuracy", "Precision", "Recall", "F1-Score"],
    datasets: pipelines.map((pipeline, index) => ({
      label: pipelineDisplayNames[pipeline],
      data: metrics[pipeline]?.error
        ? [0, 0, 0, 0]
        : [
            metrics[pipeline].accuracy,
            metrics[pipeline].precision,
            metrics[pipeline].recall,
            metrics[pipeline].f1_score,
          ],
      backgroundColor: [
        "rgba(75, 192, 192, 0.4)",
        "rgba(54, 162, 235, 0.4)",
        "rgba(255, 206, 86, 0.4)",
      ][index],
      borderColor: [
        "rgba(75, 192, 192, 1)",
        "rgba(54, 162, 235, 1)",
        "rgba(255, 206, 86, 1)",
      ][index],
      borderWidth: 1,
    })),
  };

  const comparisonBarOptions = {
    scales: { y: { beginAtZero: true, max: 1 } },
    plugins: { title: { display: true, text: "Pipeline Comparison" } },
    maintainAspectRatio: false,
    responsive: true,
  };

  const pieChartData = {
    labels: ["Knees", "Elbows", "None"],
    datasets: [
      {
        label: "Train Data",
        data: [
          dataStats.train.knees,
          dataStats.train.elbows,
          dataStats.train.none,
        ],
        backgroundColor: [
          "rgba(255, 99, 132, 0.5)",
          "rgba(54, 162, 235, 0.5)",
          "rgba(75, 192, 192, 0.5)",
        ],
      },
    ],
  };

  const pieChartOptions = {
    plugins: {
      title: { display: true, text: "Training Data Error Distribution" },
    },
    maintainAspectRatio: false,
    responsive: true,
  };

  const modelArchitectures = {};

  const modelDescriptions = {
    lstm: (
      <div className="text-sm text-gray-700">
        <h4 className="font-semibold mb-2">Inception LSTM Architecture</h4>
        <ul className="list-disc pl-5">
          <li>Takes video sequences as input.</li>
          <li>Uses an inception module to analyze movement details.</li>
          <li>Ends with a layer to predict risk levels.</li>
        </ul>
      </div>
    ),
    custom_lstm: (
      <div className="text-sm text-gray-700">
        <h3 className="font-semibold mb-2">Custom LSTM Architecture</h3>
        <ul className="list-disc pl-5">
          <li>Processes video frames one by one.</li>
          <li>Uses several LSTM layers to track movements.</li>
          <li>Finishes with a layer for frame-level risk predictions.</li>
        </ul>
      </div>
    ),
    custom_feedforward: (
      <div className="text-sm text-gray-700">
        <h4 className="font-semibold mb-2">FeedForward Architecture</h4>
        <ul className="list-disc pl-5">
          <li>Analyzes each frame separately.</li>
          <li>Uses two layers to process frame data.</li>
          <li>Provides risk predictions for each frame.</li>
        </ul>
      </div>
    ),
  };

  return (
    <div className="container mx-auto p-6 bg-gray-100 text-gray-500 min-h-screen">
      <div className="shadow-md p-3 mb-4 bg-base-100">
        <Link
          to="/"
          className="text-blue-500 hover:underline underline-offset-2 transition"
        >
          Home
        </Link>
      </div>
      <h1 className="text-3xl font-bold mb-6 text-center">
        Model Evaluation Dashboard
      </h1>

      <div className="mb-6">
        <label htmlFor="pipeline" className="mr-2 font-semibold">
          Select Pipeline:
        </label>
        <select
          id="pipeline"
          value={selectedPipeline}
          onChange={(e) => setSelectedPipeline(e.target.value)}
          className="p-2 border rounded"
        >
          {pipelines.map((pipeline) => (
            <option key={pipeline} value={pipeline}>
              {pipelineDisplayNames[pipeline]}
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-6">
        {/* 1. Model Architecture */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-bold mb-2">
            {pipelineDisplayNames[selectedPipeline]} Architecture
          </h3>
          <div className="flex items-start">
            <div className="mr-4"></div>
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 flex-1">
              {modelDescriptions[selectedPipeline]}
            </div>
          </div>
        </div>

        {/* 2. Training Data Error Distribution */}
        {/* <div className="bg-white p-6 rounded-lg shadow-lg">
          <Pie
            id="pie-chart"
            data={pieChartData}
            options={pieChartOptions}
            height={400}
          />
        </div> */}

        {/* 3. Training History */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <Line
            id="history-chart"
            data={historyChartData}
            options={historyChartOptions}
            height={400}
          />
        </div>

        {/* 4. Performance Metrics */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <Bar
            id="metric-chart"
            data={metricChartData}
            options={metricChartOptions}
            height={400}
          />
        </div>

        {/* 5. Confusion Matrix */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-bold mb-2">
            {pipelineDisplayNames[selectedPipeline]} Confusion Matrix
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <table className="w-full border-collapse border border-gray-300">
              <thead>
                <tr className="bg-gray-200">
                  <th className="border border-gray-300 p-2" colSpan="3">
                    Knees
                  </th>
                </tr>
                <tr className="bg-gray-200">
                  <th className="border border-gray-300 p-2"></th>
                  <th className="border border-gray-300 p-2">
                    Predicted: No Risk
                  </th>
                  <th className="border border-gray-300 p-2">
                    Predicted: Risk
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th className="border border-gray-300 p-2">
                    Actual: No Risk
                  </th>
                  <td className="border border-gray-300 p-2 text-center">
                    {metrics[selectedPipeline]?.error ||
                    !(
                      (metrics[selectedPipeline].confusion_matrix?.knees &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix.knees,
                        )) ||
                      (metrics[selectedPipeline].confusion_matrix_knees &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix_knees,
                        ))
                    )
                      ? "N/A"
                      : (metrics[selectedPipeline].confusion_matrix?.knees ||
                          metrics[selectedPipeline]
                            .confusion_matrix_knees)[0][0] || "N/A"}
                  </td>
                  <td className="border border-gray-300 p-2 text-center">
                    {metrics[selectedPipeline]?.error ||
                    !(
                      (metrics[selectedPipeline].confusion_matrix?.knees &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix.knees,
                        )) ||
                      (metrics[selectedPipeline].confusion_matrix_knees &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix_knees,
                        ))
                    )
                      ? "N/A"
                      : (metrics[selectedPipeline].confusion_matrix?.knees ||
                          metrics[selectedPipeline]
                            .confusion_matrix_knees)[0][1] || "N/A"}
                  </td>
                </tr>
                <tr>
                  <th className="border border-gray-300 p-2">Actual: Risk</th>
                  <td className="border border-gray-300 p-2 text-center">
                    {metrics[selectedPipeline]?.error ||
                    !(
                      (metrics[selectedPipeline].confusion_matrix?.knees &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix.knees,
                        )) ||
                      (metrics[selectedPipeline].confusion_matrix_knees &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix_knees,
                        ))
                    )
                      ? "N/A"
                      : (metrics[selectedPipeline].confusion_matrix?.knees ||
                          metrics[selectedPipeline]
                            .confusion_matrix_knees)[1][0] || "N/A"}
                  </td>
                  <td className="border border-gray-300 p-2 text-center">
                    {metrics[selectedPipeline]?.error ||
                    !(
                      (metrics[selectedPipeline].confusion_matrix?.knees &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix.knees,
                        )) ||
                      (metrics[selectedPipeline].confusion_matrix_knees &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix_knees,
                        ))
                    )
                      ? "N/A"
                      : (metrics[selectedPipeline].confusion_matrix?.knees ||
                          metrics[selectedPipeline]
                            .confusion_matrix_knees)[1][1] || "N/A"}
                  </td>
                </tr>
              </tbody>
            </table>
            <table className="w-full border-collapse border border-gray-300">
              <thead>
                <tr className="bg-gray-200">
                  <th className="border border-gray-300 p-2" colSpan="3">
                    Elbows
                  </th>
                </tr>
                <tr className="bg-gray-200">
                  <th className="border border-gray-300 p-2"></th>
                  <th className="border border-gray-300 p-2">
                    Predicted: No Risk
                  </th>
                  <th className="border border-gray-300 p-2">
                    Predicted: Risk
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th className="border border-gray-300 p-2">
                    Actual: No Risk
                  </th>
                  <td className="border border-gray-300 p-2 text-center">
                    {metrics[selectedPipeline]?.error ||
                    !(
                      (metrics[selectedPipeline].confusion_matrix?.elbows &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix.elbows,
                        )) ||
                      (metrics[selectedPipeline].confusion_matrix_elbows &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix_elbows,
                        ))
                    )
                      ? "N/A"
                      : (metrics[selectedPipeline].confusion_matrix?.elbows ||
                          metrics[selectedPipeline]
                            .confusion_matrix_elbows)[0][0] || "N/A"}
                  </td>
                  <td className="border border-gray-300 p-2 text-center">
                    {metrics[selectedPipeline]?.error ||
                    !(
                      (metrics[selectedPipeline].confusion_matrix?.elbows &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix.elbows,
                        )) ||
                      (metrics[selectedPipeline].confusion_matrix_elbows &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix_elbows,
                        ))
                    )
                      ? "N/A"
                      : (metrics[selectedPipeline].confusion_matrix?.elbows ||
                          metrics[selectedPipeline]
                            .confusion_matrix_elbows)[0][1] || "N/A"}
                  </td>
                </tr>
                <tr>
                  <th className="border border-gray-300 p-2">Actual: Risk</th>
                  <td className="border border-gray-300 p-2 text-center">
                    {metrics[selectedPipeline]?.error ||
                    !(
                      (metrics[selectedPipeline].confusion_matrix?.elbows &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix.elbows,
                        )) ||
                      (metrics[selectedPipeline].confusion_matrix_elbows &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix_elbows,
                        ))
                    )
                      ? "N/A"
                      : (metrics[selectedPipeline].confusion_matrix?.elbows ||
                          metrics[selectedPipeline]
                            .confusion_matrix_elbows)[1][0] || "N/A"}
                  </td>
                  <td className="border border-gray-300 p-2 text-center">
                    {metrics[selectedPipeline]?.error ||
                    !(
                      (metrics[selectedPipeline].confusion_matrix?.elbows &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix.elbows,
                        )) ||
                      (metrics[selectedPipeline].confusion_matrix_elbows &&
                        Array.isArray(
                          metrics[selectedPipeline].confusion_matrix_elbows,
                        ))
                    )
                      ? "N/A"
                      : (metrics[selectedPipeline].confusion_matrix?.elbows ||
                          metrics[selectedPipeline]
                            .confusion_matrix_elbows)[1][1] || "N/A"}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* 6. Pipeline Comparison */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <Bar
            id="comparison-bar"
            data={comparisonBarData}
            options={comparisonBarOptions}
            height={400}
          />
        </div>

        {/* 7. Summary */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-bold mb-4">
            {pipelineDisplayNames[selectedPipeline]} Summary
          </h3>
          <table className="w-full border-collapse bg-gray-50 rounded-lg overflow-hidden">
            <thead>
              <tr className="bg-blue-600 text-white">
                <th className="p-3 text-left font-semibold">Metric</th>
                <th className="p-3 text-left font-semibold">Value</th>
              </tr>
            </thead>
            <tbody>
              {["accuracy", "precision", "recall", "f1_score"].map((metric) => (
                <tr
                  key={metric}
                  className="border-b border-gray-200 hover:bg-gray-100"
                >
                  <td className="p-3 font-medium text-gray-800">
                    {metric.charAt(0).toUpperCase() + metric.slice(1)}
                  </td>
                  <td className="p-3 text-gray-600">
                    {metrics[selectedPipeline]?.error
                      ? "N/A"
                      : metrics[selectedPipeline][metric]?.toFixed(4) || "N/A"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
