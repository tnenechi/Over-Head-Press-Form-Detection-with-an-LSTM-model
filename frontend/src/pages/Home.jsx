import { Link } from "react-router-dom";

function Home() {
  return (
    <div className="min-h-[50vh] flex flex-col m-4 justify-between">
      <div className="shadow-md p-3 mb-4 flex items-center gap-4">
        <Link to="/" className="hover:underline underline-offset-2 transition">
          Home
        </Link>

        <a
          href="https://github.com/tnenechi/Over-Head-Press-Form-Detection-with-an-LSTM-model"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:underline underline-offset-2 transition"
        >
          About
        </a>
      </div>

      <div className="hero bg-base-200 min-h-screen">
        <div className="hero-content text-center">
          <div className="max-w-3xl">
            <h1 className="header text-7xl">
              AI-Powered Overhead Press Analysis
            </h1>
            <p className="py-6">
              Advanced AI analyzing elbow and knee alignment – better technique,
              zero injuries.
            </p>
            <div className="flex justify-center gap-6">
              <Link to="/analysis" className="btn btn-primary btn-outline">
                Analyze a Video
              </Link>
              <Link to="/dashboard" className="btn btn-primary btn-outline">
                Performance metrics
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;
