import TestYourTraffic from './TestYourTraffic';
import { useState, useEffect } from "react";
import "./App.css";

// ===== CONFIG =====
// Real backend endpoint is /predict (from src/api.py), NOT /check-traffic
const BACKEND_URL = "http://localhost:8000/predict";
const METRICS_URL = "http://localhost:8000/metrics";
const NORMAL_CSV =
  "https://raw.githubusercontent.com/Vaniya265/SIH-AI-network-traffic-anomaly-detection-using-Isolation-Forest/main/demo_normal.csv";
const HIDDEN_ATTACK_CSV =
  "https://raw.githubusercontent.com/Vaniya265/SIH-AI-network-traffic-anomaly-detection-using-Isolation-Forest/main/demo_hidden_attack.csv";

// Real, tested number (PDF spec se) — total test set size fixed hai,
// isliye ye static rehta hai, backend se fetch karne ki zaroorat nahi.
const TOTAL_SAMPLES_PROCESSED = 22544;

async function loadCSVAsRows(url) {
  const res = await fetch(url);
  const text = await res.text();
  const [headerLine, ...lines] = text.trim().split("\n");
  const headers = headerLine.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      const v = values[i];
      row[h] = isNaN(v) ? v : Number(v);
    });
    return row;
  });
}

async function checkTraffic(row) {
  const response = await fetch(BACKEND_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(row),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Backend error ${response.status}: ${detail}`);
  }
  return await response.json();
}

// Backend's detection_source -> our two-layer labels
function mapDetectionSource(source) {
  if (source === "Isolation Forest") return "Behavior Layer";
  if (source === "Random Forest Safety Net") return "Pattern Layer";
  return null; // "None" or missing
}

function App() {
  const [score, setScore] = useState(18);
  const [verdict, setVerdict] = useState("SAFE"); // backend sends UPPERCASE
  const [threatLevel, setThreatLevel] = useState("LOW"); // now comes from backend's risk_level
  const [detectedBy, setDetectedBy] = useState(null);
  const [rfPrediction, setRfPrediction] = useState(null);
  const [reasons, setReasons] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentLabel, setCurrentLabel] = useState("");
  const [error, setError] = useState(null);
  const [safeCount, setSafeCount] = useState(0);
  const [anomalyCount, setAnomalyCount] = useState(0);

  // Dynamic metrics — backend ke /metrics endpoint se aate hain
  const [catchRate, setCatchRate] = useState(null);
  const [falseAlarmRate, setFalseAlarmRate] = useState(null);

  useEffect(() => {
    fetch(METRICS_URL)
      .then((res) => res.json())
      .then((data) => {
        setCatchRate(data.catch_rate);
        setFalseAlarmRate(data.false_alarm_rate);
      })
      .catch((err) => console.error("Could not load metrics:", err));
  }, []);

  const runSimulation = async () => {
    if (isRunning) return;

    setIsRunning(true);
    setError(null);
    setScore(18);
    setVerdict("SAFE");
    setThreatLevel("LOW");
    setDetectedBy(null);
    setRfPrediction(null);
    setReasons([]);
    setSafeCount(0);
    setAnomalyCount(0);

    try {
      const normalRows = await loadCSVAsRows(NORMAL_CSV);
      const hiddenAttackRows = await loadCSVAsRows(HIDDEN_ATTACK_CSV);

      const demoSequence = [
        ...normalRows.slice(0, 2).map((r) => ({ ...r, __label: "Normal traffic" })),
        ...hiddenAttackRows.map((r) => ({ ...r, __label: "Unseen attack (hidden from training)" })),
      ];

      for (const row of demoSequence) {
        const { __label, ...payload } = row;
        setCurrentLabel(__label);

        // result: { status, verdict, risk_score, risk_level, detection_source, rf_prediction, reasons }
        const result = await checkTraffic(payload);
        const {
          status,
          verdict: resultVerdict,
          risk_score,
          risk_level,
          detection_source,
          rf_prediction,
          reasons: resultReasons,
        } = result;

        if (status && status !== "success") {
          console.error("Backend returned error status:", result);
        }

        setScore(risk_score);
        setVerdict(resultVerdict); // already "SAFE" / "SUSPICIOUS" / "COMPROMISED"
        setThreatLevel(risk_level || "LOW"); // straight from model, not locally computed
        setDetectedBy(mapDetectionSource(detection_source));
        setRfPrediction(rf_prediction ?? null);
        setReasons(resultReasons || []);

        if (resultVerdict === "SAFE") {
          setSafeCount((c) => c + 1);
        } else {
          setAnomalyCount((c) => c + 1);
        }

        await new Promise((r) => setTimeout(r, 1500));
      }
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setIsRunning(false);
      setCurrentLabel("");
    }
  };

  const isDanger = verdict === "COMPROMISED";
  const isSuspicious = verdict === "SUSPICIOUS";

  return (
    <div className="app">
      {/* ================= HEADER / BRAND BOX ================= */}
      <header className="top-header">
        <div className="brand">
          <div className="brand-logo">🛡</div>
          <div>
            <h1>NetSentinel</h1>
            <p>AI-Powered Network Security</p>
          </div>
        </div>

        <div className="header-right">
          <span>◐</span>
          <span className="icon-group">
            <span className="nav-active">Dashboard</span>
            <span>🔔</span>
          </span>
          <div className="profile">
            <div className="profile-avatar">S</div>
            <div>
              <strong>Security Admin</strong>
              <small>AI Security</small>
            </div>
          </div>
        </div>
      </header>

      {/* ================= TOP CARDS ================= */}
      <section className="stats-grid">
        <div className="stat-card stat-card-wave">
          <div className="stat-icon">🛡</div>
          <span>RISK SCORE</span>
          <strong>{score}</strong>
          <small className={isDanger ? "danger" : isSuspicious ? "" : "safe"}>
            {verdict}
          </small>
          <svg className="stat-wave" viewBox="0 0 200 36" preserveAspectRatio="none">
            <path
              key={verdict}
              className={isDanger ? "wave-path danger-wave" : "wave-path"}
              d="M0,22 Q15,8 30,18 T60,16 T90,22 T120,12 T150,20 T180,10 T200,18"
              fill="none"
              strokeWidth="2"
            />
          </svg>
        </div>

        <div className="stat-card">
          <div className="stat-icon">◎</div>
          <span>THREAT LEVEL</span>
          <strong className={isDanger ? "danger" : "safe"}>{threatLevel}</strong>
          <small>Current network risk</small>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🧠</div>
          <span>DETECTED BY</span>
          <strong style={{ fontSize: "18px" }}>{detectedBy || "—"}</strong>
          <small>
            {rfPrediction
              ? `RF prediction: ${rfPrediction}`
              : detectedBy
              ? "Model layer that flagged this"
              : "No flag yet"}
          </small>
        </div>

        <div className="stat-card">
          <div className="stat-icon">▤</div>
          <span>SAMPLES PROCESSED</span>
          <strong>{TOTAL_SAMPLES_PROCESSED.toLocaleString()}</strong>
          <small>NSL-KDD test set</small>
        </div>
      </section>

      {/* ================= DETECTION LAYERS STATUS STRIP ================= */}
      <section className="panel status-strip">
        <div className="status-item">
          <span className="status-dot active"></span>
          <span>Behavior Layer (Isolation Forest) — Active</span>
        </div>
        <div className="status-item">
          <span className="status-dot active"></span>
          <span>Pattern Layer (Random Forest) — Active</span>
        </div>
      </section>

      {/* ================= TRAFFIC OVERVIEW + WHY FLAGGED ================= */}
      <section className="dashboard-row">
        <div className="panel">
          <div className="panel-header">
            <div>
              <h2>Traffic Overview</h2>
              <p>Safe vs anomaly counts from this session</p>
            </div>
          </div>

          <div className="safe-anomaly-chart">
            <div className="sa-badge-col">
              <div className="sa-icon-badge safe-badge">🛡</div>
              <strong className="safe">{safeCount}</strong>
              <small>Safe</small>
            </div>

            <div className="sa-badge-col">
              <div className="sa-icon-badge anomaly-badge">⚠</div>
              <strong className="danger">{anomalyCount}</strong>
              <small>Anomaly</small>
            </div>
          </div>
        </div>

        <div className="panel panel-flagged">
          <div className="panel-header">
            <div>
              <h2>Why flagged?</h2>
              <p>Plain-language reasons for the current verdict</p>
            </div>
          </div>

          {reasons.length > 0 ? (
            <ul className="flagged-list">
              {reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          ) : (
            <p className="flagged-empty">
              {verdict === "SAFE"
                ? "Traffic matches normal behavior patterns — no reasons needed."
                : "Run the simulation to see the reasons behind a flagged event."}
            </p>
          )}
        </div>
      </section>

      {/* ================= FOOTER STAT (DYNAMIC — from backend /metrics) ================= */}
      <section className="panel footer-stat">
        <strong>{catchRate !== null ? `${catchRate}%` : "—"}</strong> catch rate on unseen attacks &nbsp;·&nbsp;{" "}
        <strong>{falseAlarmRate !== null ? `${falseAlarmRate}%` : "—"}</strong> false alarm rate
      </section>

      {/* ================= TEST YOUR OWN TRAFFIC (judge custom input) ================= */}
      <TestYourTraffic
        onResult={(result) => {
          setScore(result.risk_score);
          setVerdict(result.verdict || result.status);
          setThreatLevel(result.risk_level || "LOW");
          setReasons(result.reasons || []);
          setDetectedBy(mapDetectionSource(result.detection_source));
          setRfPrediction(result.rf_prediction ?? null);
        }}
      />

      {/* ================= SIMULATION ================= */}
      {currentLabel && (
        <p className="testing-label">
          Testing: <strong>{currentLabel}</strong>
        </p>
      )}

      {error && (
        <p className="error-message">
          {error} — check backend is running on {BACKEND_URL}
        </p>
      )}

      <button className="simulation-button" onClick={runSimulation} disabled={isRunning}>
        {isRunning ? "● SIMULATION RUNNING..." : "▶ RUN LIVE SIMULATION"}
      </button>
    </div>
  );
}

export default App;
