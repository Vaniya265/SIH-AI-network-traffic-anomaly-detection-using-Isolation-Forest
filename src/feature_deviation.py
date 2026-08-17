"""
feature_deviation.py
Person 2 (Sweta) - "WHY was it flagged?" module

Matched exactly to Vaniya's repo:
  Vaniya265/SIH-AI-network-traffic-anomaly-detection-using-Isolation-Forest

Given one RAW traffic row (same dict shape her api.py's TrafficData expects)
+ baseline_stats.json (built from her real normal KDDTrain+ data),
this returns the top N features that deviate most from normal, as
plain-language reasons for the dashboard's explainability panel.

IMPORTANT: pass the RAW input dict here (same values sent to /predict),
NOT the one-hot-encoded / scaled version her model.pkl actually sees.
Z-scores need real feature values to mean anything to a human.
"""
import json

FEATURE_LABELS = {
    "duration": "Connection duration",
    "src_bytes": "Source bytes (data sent)",
    "dst_bytes": "Destination bytes (data received)",
    "land": "Same host/port connection flag",
    "wrong_fragment": "Wrong-fragment packet count",
    "urgent": "Urgent packet count",
    "hot": "Hot indicator count",
    "num_failed_logins": "Failed login attempts",
    "logged_in": "Login status",
    "num_compromised": "Compromised condition count",
    "root_shell": "Root shell access",
    "su_attempted": "'su root' attempts",
    "num_root": "Root accesses count",
    "num_file_creations": "File creation count",
    "num_shells": "Shell prompts opened",
    "num_access_files": "Access-control file operations",
    "num_outbound_cmds": "Outbound commands in FTP session",
    "is_host_login": "Host login flag",
    "is_guest_login": "Guest login flag",
    "count": "Connection count to same host (last 2s)",
    "srv_count": "Connection count to same service (last 2s)",
    "serror_rate": "SYN error rate",
    "srv_serror_rate": "Service SYN error rate",
    "rerror_rate": "REJ error rate",
    "srv_rerror_rate": "Service REJ error rate",
    "same_srv_rate": "Same-service connection rate",
    "diff_srv_rate": "Different-service connection rate",
    "srv_diff_host_rate": "Different-host service rate",
    "dst_host_count": "Destination host connection count",
    "dst_host_srv_count": "Destination host service count",
    "dst_host_same_srv_rate": "Destination host same-service rate",
    "dst_host_diff_srv_rate": "Destination host different-service rate",
    "dst_host_same_src_port_rate": "Destination host same-source-port rate",
    "dst_host_srv_diff_host_rate": "Destination host service/different-host rate",
    "dst_host_serror_rate": "Destination host SYN error rate",
    "dst_host_srv_serror_rate": "Destination host service SYN error rate",
    "dst_host_rerror_rate": "Destination host REJ error rate",
    "dst_host_srv_rerror_rate": "Destination host service REJ error rate",
}


class FeatureDeviationEngine:
    def __init__(self, baseline_path="baseline_stats.json"):
        with open(baseline_path, "r") as f:
            self.baseline = json.load(f)

    def _label(self, feature):
        return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())

    def _direction_phrase(self, feature, z):
        label = self._label(feature)
        severity = "far" if abs(z) >= 3 else "significantly"
        direction = "above" if z > 0 else "below"
        return f"{label} is {severity} {direction} normal"

    def get_deviations(self, row: dict):
        """List of (feature, z_score), sorted by |z_score| descending."""
        deviations = []
        for feature, stat in self.baseline.items():
            if feature not in row:
                continue
            try:
                value = float(row[feature])
            except (TypeError, ValueError):
                continue
            z = (value - stat["mean"]) / stat["std"]
            deviations.append((feature, z))
        deviations.sort(key=lambda x: abs(x[1]), reverse=True)
        return deviations

    def get_reasons(self, row: dict, top_n=3, z_threshold=1.0):
        """
        Main function -> plain-language reasons for the dashboard.
        Only includes features that genuinely deviate (|z| > threshold).
        Falls back to the single top feature if nothing crosses the
        threshold, so the API never returns an empty reasons list.
        """
        deviations = self.get_deviations(row)
        significant = [d for d in deviations if abs(d[1]) >= z_threshold]
        chosen = significant[:top_n] if significant else deviations[:1]
        return [self._direction_phrase(f, z) for f, z in chosen]

    def get_reasons_with_scores(self, row: dict, top_n=3, z_threshold=1.0):
        """Same as get_reasons but includes raw z-scores — useful for testing/demo prep."""
        deviations = self.get_deviations(row)
        significant = [d for d in deviations if abs(d[1]) >= z_threshold]
        chosen = significant[:top_n] if significant else deviations[:1]
        return [
            {
                "feature": f,
                "label": self._label(f),
                "z_score": round(z, 2),
                "reason": self._direction_phrase(f, z),
            }
            for f, z in chosen
        ]
