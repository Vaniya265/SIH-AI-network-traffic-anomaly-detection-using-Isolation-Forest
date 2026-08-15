import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. LOAD DETECTION RESULTS
# ==========================================

results = pd.read_csv(
    "data/detection_results.csv"
)


# ==========================================
# 2. CONVERT TRUE LABELS
# ==========================================

# normal = normal
# everything else = attack

y_true = results["actual"].apply(
    lambda x: "normal" if x == "normal" else "attack"
)


# ==========================================
# 3. CONVERT PREDICTIONS
# ==========================================

y_pred = results["predicted"].apply(
    lambda x: "normal" if x == "normal" else "attack"
)


# ==========================================
# 4. ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred,
    pos_label="attack"
)


recall = recall_score(
    y_true,
    y_pred,
    pos_label="attack"
)


f1 = f1_score(
    y_true,
    y_pred,
    pos_label="attack"
)


# ==========================================
# 5. PRINT METRICS
# ==========================================

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ==========================================
# 6. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=["normal", "attack"]
)


print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print("                 Predicted")
print("              Normal  Attack")

print(
    f"Actual Normal   {cm[0][0]:6d}  {cm[0][1]:6d}"
)

print(
    f"Actual Attack   {cm[1][0]:6d}  {cm[1][1]:6d}"
)


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_true,
        y_pred
    )
)