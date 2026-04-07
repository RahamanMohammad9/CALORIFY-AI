import os
import re

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

ACCURACY_PATH = os.path.join(RESULTS_DIR, "accuracy.txt")
ADVANCED_PATH = os.path.join(RESULTS_DIR, "advanced_metrics.txt")
PER_CLASS_AUC_PATH = os.path.join(RESULTS_DIR, "per_class_auc.csv")

SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, "leaderboard_summary.csv")
SUMMARY_MD_PATH = os.path.join(RESULTS_DIR, "leaderboard_summary.md")


def parse_key_value_file(path: str) -> dict:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _to_float_maybe(s: str):
    if s is None:
        return None
    cleaned = re.sub(r"[%\s]", "", str(s))
    try:
        return float(cleaned)
    except ValueError:
        return None


def main():
    acc = parse_key_value_file(ACCURACY_PATH)
    adv = parse_key_value_file(ADVANCED_PATH)

    top_classes = []
    bottom_classes = []
    if os.path.exists(PER_CLASS_AUC_PATH):
        auc_df = pd.read_csv(PER_CLASS_AUC_PATH)
        if not auc_df.empty and "roc_auc" in auc_df.columns and "class" in auc_df.columns:
            top = auc_df.sort_values("roc_auc", ascending=False).head(5)
            bottom = auc_df.sort_values("roc_auc", ascending=True).head(5)
            top_classes = [f"{r['class']} ({r['roc_auc']:.3f})" for _, r in top.iterrows()]
            bottom_classes = [f"{r['class']} ({r['roc_auc']:.3f})" for _, r in bottom.iterrows()]

    summary_row = {
        "accuracy_pct": _to_float_maybe(acc.get("Test Accuracy")),
        "macro_f1": _to_float_maybe(acc.get("Macro F1 Score")),
        "weighted_f1": _to_float_maybe(acc.get("Weighted F1 Score")),
        "roc_auc_micro": _to_float_maybe(adv.get("ROC AUC (micro)")),
        "pr_auc_micro": _to_float_maybe(adv.get("PR AUC / MUI (micro)")),
        "ece_10_bins": _to_float_maybe(adv.get("ECE (10 bins)")),
        "brier_macro": _to_float_maybe(adv.get("Brier score (macro over classes)")),
        "top5_classes_by_roc_auc": " | ".join(top_classes),
        "bottom5_classes_by_roc_auc": " | ".join(bottom_classes),
    }

    pd.DataFrame([summary_row]).to_csv(SUMMARY_CSV_PATH, index=False)

    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Model Leaderboard Summary\n\n")
        f.write(f"- Accuracy: {summary_row['accuracy_pct']:.2f}%\n")
        f.write(f"- Macro F1: {summary_row['macro_f1']:.4f}\n")
        f.write(f"- Weighted F1: {summary_row['weighted_f1']:.4f}\n")
        f.write(f"- ROC AUC (micro): {summary_row['roc_auc_micro']:.4f}\n")
        f.write(f"- PR AUC / MUI (micro): {summary_row['pr_auc_micro']:.4f}\n")
        f.write(f"- ECE (10 bins): {summary_row['ece_10_bins']:.4f}\n")
        f.write(f"- Brier score (macro): {summary_row['brier_macro']:.4f}\n\n")
        if top_classes:
            f.write("## Top 5 Classes by ROC-AUC\n")
            for item in top_classes:
                f.write(f"- {item}\n")
            f.write("\n")
        if bottom_classes:
            f.write("## Bottom 5 Classes by ROC-AUC\n")
            for item in bottom_classes:
                f.write(f"- {item}\n")

    print(f"Saved: {SUMMARY_CSV_PATH}")
    print(f"Saved: {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()
