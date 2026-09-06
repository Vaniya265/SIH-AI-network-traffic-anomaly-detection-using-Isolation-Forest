import { useState } from "react";

const VISIBLE_FIELDS = [
  "duration",
  "src_bytes",
  "dst_bytes",
  "count",
  "srv_count",
  "serror_rate",
  "srv_serror_rate",
  "rerror_rate",
  "num_failed_logins",
  "logged_in",
  "same_srv_rate",
  "dst_host_serror_rate",
  "dst_host_same_srv_rate"
];

const DEFAULT_VALUES = {
  duration: 0,
  src_bytes: 0,
  dst_bytes: 0,
  count: 0,
  srv_count: 0,
  serror_rate: 0,
  srv_serror_rate: 0,
  rerror_rate: 0,
  num_failed_logins: 0,
  logged_in: 0,
  same_srv_rate: 0,
  dst_host_serror_rate: 0,
  dst_host_same_srv_rate: 0
};

export default function TestYourTraffic({ onResult }) {
  const [values, setValues] = useState(DEFAULT_VALUES);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (field, val) => {
    setValues((prev) => ({
      ...prev,
      [field]: Number(val)
    }));
  };

  const analyze = async () => {
    console.log("🔥 ANALYZE BUTTON CLICKED");

    setLoading(true);
    setError("");

    try {
      // ==========================================
      // STEP 1: Load baseline_stats.json
      // ==========================================

      console.log("📁 Loading baseline_stats.json...");

      const baselineRes = await fetch("/baseline_stats.json");

      console.log(
        "📁 baseline_stats.json status:",
        baselineRes.status
      );

      if (!baselineRes.ok) {
        throw new Error(
          `baseline_stats.json could not be loaded. Status: ${baselineRes.status}`
        );
      }

      const baseline = await baselineRes.json();

      console.log("✅ Baseline loaded:", baseline);

      // ==========================================
      // STEP 2: Automatically fill remaining fields
      // ==========================================

      const autoFilled = {};

      for (const [key, stat] of Object.entries(baseline)) {
        if (!VISIBLE_FIELDS.includes(key)) {
          /*
            Rate fields → decimal/float
            Other fields → integer
          */

          if (key.includes("rate")) {
            autoFilled[key] = Number(stat.mean);
          } else {
            autoFilled[key] = Math.round(Number(stat.mean));
          }
        }
      }

      console.log("🤖 Auto-filled features:", autoFilled);

      // ==========================================
      // STEP 3: Create complete payload
      // ==========================================

      const fullPayload = {
        protocol_type: "tcp",
        service: "http",
        flag: "SF",

        ...autoFilled,

        // Judge-entered values overwrite defaults
        ...values
      };

      console.log("📦 FULL PAYLOAD SENT TO BACKEND:");
      console.log(fullPayload);

      // ==========================================
      // STEP 4: Send to SAME /predict endpoint
      // ==========================================

      console.log("🚀 Sending request to /predict...");

      const response = await fetch(
        "http://localhost:8000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(fullPayload)
        }
      );

      console.log(
        "🚀 /predict response status:",
        response.status
      );

      // ==========================================
      // STEP 5: Handle backend error
      // ==========================================

      if (!response.ok) {
        const detail = await response.text();

        console.error(
          "❌ Backend rejected request:",
          response.status,
          detail
        );

        setError(
          `Backend Error ${response.status}: ${detail}`
        );

        return;
      }

      // ==========================================
      // STEP 6: Get prediction result
      // ==========================================

      const result = await response.json();

      console.log("🎯 PREDICTION RESULT:");
      console.log(result);

      // ==========================================
      // STEP 7: Send result to App.jsx
      // ==========================================

      onResult(result);

      console.log("✅ Dashboard updated successfully!");

    } catch (err) {
      console.error("❌ Prediction failed:", err);

      setError(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        padding: "20px",
        border: "1px solid #333",
        borderRadius: "10px",
        marginTop: "20px"
      }}
    >
      <h3>TEST YOUR OWN TRAFFIC</h3>

      <p style={{ fontSize: "13px", opacity: 0.7 }}>
        Enter traffic values below and send them to the same AI
        prediction model used by the live simulation.
      </p>

      {/* ==========================================
          INPUT FIELDS
          ========================================== */}

      {VISIBLE_FIELDS.map((field) => (
        <div
          key={field}
          style={{
            marginBottom: "12px"
          }}
        >
          <label
            style={{
              display: "block",
              fontSize: "12px",
              marginBottom: "4px"
            }}
          >
            {field}
          </label>

          <input
            type="number"
            step="any"
            value={values[field]}
            onChange={(e) =>
              handleChange(field, e.target.value)
            }
            style={{
              width: "100%",
              padding: "8px",
              boxSizing: "border-box"
            }}
          />
        </div>
      ))}

      {/* ==========================================
          ERROR MESSAGE
          ========================================== */}

      {error && (
        <div
          style={{
            marginTop: "12px",
            padding: "10px",
            borderRadius: "6px",
            background: "#3a1111",
            color: "#ff6b6b",
            fontSize: "13px",
            wordBreak: "break-word"
          }}
        >
          ❌ {error}
        </div>
      )}

      {/* ==========================================
          ANALYZE BUTTON
          ========================================== */}

      <button
        onClick={analyze}
        disabled={loading}
        style={{
          marginTop: "15px",
          padding: "10px 18px",
          cursor: loading ? "not-allowed" : "pointer"
        }}
      >
        {loading
          ? "Analyzing..."
          : "ANALYZE TRAFFIC"}
      </button>
    </div>
  );
}