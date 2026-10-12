"""
Student Placement & Salary Prediction — Leak-Free Pipeline
EDA → Viz → Preprocessing → FE → 5 Models → Comparison → ROC → Deploy
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_PATH   = "student_placement_salary_elite_v2.csv"
OUTPUT_DIR  = "outputs"
MODEL_DIR   = "models"
RANDOM_STATE = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==================================================================
# 1. LOAD + FEATURE ENGINEERING
# ==================================================================
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape}")
    return df


def add_features(df):
    """Domain-driven features that boost accuracy legitimately."""
    df = df.copy()
    df["skill_total"]  = df[["python_skill","dsa_skill","ml_skill","web_dev_skill"]].sum(axis=1)
    df["coding_dsa"]   = df["coding_score"] * df["dsa_skill"]
    df["cgpa_tier"]    = df["cgpa"] / df["college_tier"]
    df["proj_intern"]  = df["projects"] + df["internships"] * 2
    df["resume_comm"]  = df["resume_score"] * df["communication_score"]
    df["backlog_pen"]  = df["backlogs"] * -1
    df["aptitude_cgpa"]= df["aptitude_score"] * df["cgpa"] / 10
    return df


# ==================================================================
# 2. EDA (compact)
# ==================================================================
def run_eda(df):
    print("=" * 70, "\nEDA\n", "=" * 70)
    print(df.describe().T)
    print("\nMissing:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\nTarget:\n", df["placed"].value_counts(normalize=True))
    with open(f"{OUTPUT_DIR}/eda_report.txt", "w") as f:
        f.write(df.describe().T.to_string())


# ==================================================================
# 3. VISUALIZATION
# ==================================================================
def run_visualizations(df):
    placed = df[df["placed"] == 1]

    plots = {
        "01_target": (sns.countplot, dict(x="placed", data=df)),
        "02_salary": (sns.histplot, dict(x=placed["salary_lpa"], kde=True, bins=40)),
        "03_cgpa_salary": (sns.scatterplot, dict(x="cgpa", y="salary_lpa", hue="company_type", data=placed, alpha=0.6)),
        "04_salary_branch": (sns.boxplot, dict(x="branch", y="salary_lpa", data=placed)),
    }
    for name, (fn, kw) in plots.items():
        plt.figure()
        fn(**kw)
        plt.title(name)
        plt.savefig(f"{OUTPUT_DIR}/{name}.png", bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(14, 10))
    sns.heatmap(df.select_dtypes(np.number).corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{OUTPUT_DIR}/05_corr.png", bbox_inches="tight")
    plt.close()
    print(f"✅ Visualizations saved → {OUTPUT_DIR}/")


# ==================================================================
# 4. PREPROCESSING
# ==================================================================
def build_preprocessor(X):
    num_cols = X.select_dtypes(np.number).columns.tolist()
    cat_cols = X.select_dtypes("object").columns.tolist()

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
    ])
    return pre


def prepare_classification_data(df):
    """
    LEAK-FREE classification setup.
    Drops: student_id, company_type, job_role, salary_lpa
    (all are post-placement info → leak the target)
    """
    df = add_features(df)
    df = df.drop(columns=["student_id", "company_type", "job_role", "salary_lpa"])
    X = df.drop(columns=["placed"])
    y = df["placed"]
    return X, y


def prepare_regression_data(df):
    """Salary regression — only placed students."""
    placed = df[df["placed"] == 1].copy()
    placed = add_features(placed)
    placed = placed.drop(columns=["student_id", "placed", "company_type", "job_role"])
    X = placed.drop(columns=["salary_lpa"])
    y = placed["salary_lpa"]
    return X, y


# ==================================================================
# 5. CLASSIFICATION — 5 MODELS
# ==================================================================
def train_classification(X, y):
    print("\n" + "=" * 70, "\nCLASSIFICATION — 5 MODELS\n", "=" * 70)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pre = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=12,
                                                      random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                             eval_metric="logloss", random_state=RANDOM_STATE,
                                             n_jobs=-1),
        "LightGBM":            LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                              random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    }

    results, trained = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_te, y_pred),
            "F1":       f1_score(y_te, y_pred),
            "ROC-AUC":  roc_auc_score(y_te, y_prob),
        }
        results.append(row)
        trained[name] = pipe
        print(f"{name:22s}  Acc={row['Accuracy']:.4f}  F1={row['F1']:.4f}  AUC={row['ROC-AUC']:.4f}")

    res = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    res.to_csv(f"{OUTPUT_DIR}/classification_comparison.csv", index=False)

    # Comparison bar plot
    res.set_index("Model")[["Accuracy", "F1", "ROC-AUC"]].plot(kind="bar", figsize=(10, 5))
    plt.title("Classification — Model Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_model_comparison.png", bbox_inches="tight")
    plt.close()

    # ROC curves
    plt.figure()
    for name, pipe in trained.items():
        y_prob = pipe.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curves — All Classifiers")
    plt.legend(loc="lower right")
    plt.savefig(f"{OUTPUT_DIR}/10_roc_curves.png", bbox_inches="tight")
    plt.close()

    # Best model
    best_name = res.iloc[0]["Model"]
    best_pipe = trained[best_name]

    # Confusion matrix
    cm = confusion_matrix(y_te, best_pipe.predict(X_te))
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.savefig(f"{OUTPUT_DIR}/11_cm.png", bbox_inches="tight")
    plt.close()

    # Feature importance
    _plot_importance(best_pipe, X, "14_cls_importance.png")

    joblib.dump(best_pipe, f"{MODEL_DIR}/best_classifier.pkl")
    print(f"\n✅ Best classifier: {best_name} → saved")
    return res, best_pipe, best_name, X_te, y_te


def _plot_importance(pipe, X, fname):
    clf = pipe.named_steps["clf"]
    pre = pipe.named_steps["pre"]
    names = pre.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).ravel()
    else:
        return

    idx = np.argsort(imp)[::-1][:20]
    plt.figure(figsize=(10, 8))
    sns.barplot(x=imp[idx], y=np.array(names)[idx])
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{fname}", bbox_inches="tight")
    plt.close()


# ==================================================================
# 6. REGRESSION — SALARY PREDICTION
# ==================================================================
def train_regression(df):
    print("\n" + "=" * 70, "\nREGRESSION — SALARY\n", "=" * 70)
    X, y = prepare_regression_data(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pre = build_preprocessor(X)

    models = {
        "Linear":        Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12,
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost":       XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                       random_state=RANDOM_STATE, n_jobs=-1),
    }

    results, trained = [], {}
    for name, reg in models.items():
        pipe = Pipeline([("pre", pre), ("reg", reg)]).fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        row = {
            "Model": name,
            "MAE":  mean_absolute_error(y_te, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_te, y_pred)),
            "R2":   r2_score(y_te, y_pred),
        }
        results.append(row); trained[name] = pipe
        print(f"{name:15s}  MAE={row['MAE']:.3f}  RMSE={row['RMSE']:.3f}  R²={row['R2']:.4f}")

    res = pd.DataFrame(results).sort_values("R2", ascending=False)
    res.to_csv(f"{OUTPUT_DIR}/regression_comparison.csv", index=False)

    # Actual vs Predicted
    best_name = res.iloc[0]["Model"]
    best_pipe = trained[best_name]
    y_pred = best_pipe.predict(X_te)

    plt.figure()
    plt.scatter(y_te, y_pred, alpha=0.5, color="teal")
    plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "r--")
    plt.xlabel("Actual Salary"); plt.ylabel("Predicted Salary")
    plt.title(f"Actual vs Predicted — {best_name}")
    plt.savefig(f"{OUTPUT_DIR}/12_actual_vs_pred.png", bbox_inches="tight")
    plt.close()

    joblib.dump(best_pipe, f"{MODEL_DIR}/best_regressor.pkl")
    print(f"\n✅ Best regressor: {best_name} → saved")
    return res, best_pipe, best_name


# ==================================================================
# 7. MAIN
# ==================================================================
def main():
    df = load_data()
    run_eda(df)
    run_visualizations(df)

    X_cls, y_cls = prepare_classification_data(df)
    cls_res, best_cls, best_cls_name, X_te, y_te = train_classification(X_cls, y_cls)

    reg_res, best_reg, best_reg_name = train_regression(df)

    # Sample prediction
    print("\n" + "=" * 70, "\nSAMPLE PREDICTION\n", "=" * 70)
    sample = X_te.iloc[[0]]
    placed = best_cls.predict(sample)[0]
    prob   = best_cls.predict_proba(sample)[0][1]
    print(f"Placed={placed}  Prob={prob:.3f}")
    if placed == 1:
        salary = best_reg.predict(sample)[0]
        print(f"Predicted Salary: ₹{salary:.2f} LPA")

    # Summary
    summary = f"""
================= PROJECT SUMMARY =================
Best Classifier: {best_cls_name}
Best Regressor : {best_reg_name}
---------------------------------------------------
Classification Results:
{cls_res.to_string(index=False)}
---------------------------------------------------
Regression Results:
{reg_res.to_string(index=False)}
===================================================
"""
    with open(f"{OUTPUT_DIR}/final_summary.txt", "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()