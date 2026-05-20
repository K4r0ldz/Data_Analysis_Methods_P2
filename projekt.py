import math
import os
from concurrent.futures import ProcessPoolExecutor, wait

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np


"""
    Osoba A
"""

# Wczytuje dane z pliku CSV
# Usuwa kolumny typu leakage/ID/błędy/pozycja,
# Zwraca macierz cech X z nazwanymi kolumnami
# Zwraca wektor celu Y, gdzie 1 oznacza CONFIRMED, 0 FALSE POSITIVE
def process_kepler_data(filepath):

    df = pd.read_csv(filepath)

    cols_to_drop = [
        "rowid",
        "kepid",
        "kepoi_name",
        "kepler_name",
        "koi_tce_delivname",
        "koi_pdisposition",
        "koi_score",
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec",
        "ra",
        "dec",
    ]

    err_cols = [c for c in df.columns if "_err1" in c or "_err2" in c]
    df = df.drop(columns=cols_to_drop + err_cols)

    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()

    df["koi_disposition"] = df["koi_disposition"].map(
        {"CONFIRMED": 1, "FALSE POSITIVE": 0}
    )

    rename_map = {
        "koi_disposition": "target",
        "koi_period": "okres_orbitalny",
        "koi_duration": "czas_trwania",
        "koi_depth": "glebokosc_tranzytu",
        "koi_prad": "promien_planety",
        "koi_teq": "temperatura_rownowagi",
        "koi_insol": "insolacja",
        "koi_model_snr": "stosunek_sygnal_szum",
        "koi_steff": "temperatura_efektywna",
        "koi_slogg": "log_grawitacji",
        "koi_srad": "promien_gwiazdy",
        "koi_impact": "parametr_zderzenia",
        "koi_kepmag": "jasnosc_kepler",
    }

    df = df.rename(columns=rename_map)

    X = df.drop(columns=["target"])
    Y = df["target"]

    return X, Y


# Oblicza statystyki opisowe
# Zapisuje w pliku tables/statystyki_opisowe.csv
def save_descriptive_statistics(df, filepath="tables/statystyki_opisowe.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    desc = df.describe().T

    additional_stats = pd.DataFrame(
        {
            "skośność": df.apply(lambda x: stats.skew(x.dropna())),
            "kurtoza": df.apply(lambda x: stats.kurtosis(x.dropna())),
            "IQR": df.quantile(0.75) - df.quantile(0.25),
            "brak_danych_%": df.isnull().mean() * 100,
        }
    )

    stats_table = pd.concat([desc, additional_stats], axis=1)
    stats_table.to_csv(filepath)
    return stats_table


# Wyświetla statystyki opisowe w czytelnych porcjach
def pretty_print_descriptive_statistics(stats_table, columns_per_batch=5):
    columns = stats_table.columns.tolist()
    for i in range(0, len(columns), columns_per_batch):
        batch = columns[i : i + columns_per_batch]
        print(stats_table[batch].round(3).to_string())
        print()
        print()


def _plot_compact_hist(df, num_cols):
    cols = 3
    rows = math.ceil(len(num_cols) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, kde=True, ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("")

    for i in range(len(num_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    fig.savefig("figures/compact_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_compact_box(df, num_cols):
    cols = 3
    rows = math.ceil(len(num_cols) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(data=df, x="target", y=col, ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_ylabel("")

    for i in range(len(num_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    fig.savefig("figures/compact_box.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_compact_violin(df, num_cols):
    cols = 3
    rows = math.ceil(len(num_cols) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.violinplot(data=df, x="target", y=col, inner="quartile", ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_ylabel("")

    for i in range(len(num_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    fig.savefig("figures/compact_violin.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_corr(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0)
    plt.title("Macierz korelacji")
    plt.savefig("figures/correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_count(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=df)
    plt.title("Balans klas (Target)")
    plt.savefig("figures/countplot_target.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_pair(df):
    key_features = [
        "okres_orbitalny",
        "glebokosc_tranzytu",
        "promien_planety",
        "insolacja",
        "target",
    ]
    available_features = [f for f in key_features if f in df.columns]
    if len(available_features) > 1:
        sns.pairplot(df[available_features], hue="target")
        plt.savefig("figures/pairplot_key_features.png", dpi=300, bbox_inches="tight")
        plt.close()


# Generuje i zapisuje wykresy dla eksploracyjnej analizy danych.
# Zmienne takie jak okres_orbitalny, insolacja, glebokosc_tranzytu i promien_planety
# zazwyczaj wykazują silną prawoskośność.
def generate_visualizations(df):
    os.makedirs("figures", exist_ok=True)
    num_cols = (
        df.select_dtypes(include=["number"])
        .columns.drop("target", errors="ignore")
        .tolist()
    )

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_plot_compact_hist, df, num_cols),
            executor.submit(_plot_compact_box, df, num_cols),
            executor.submit(_plot_compact_violin, df, num_cols),
            executor.submit(_plot_corr, df),
            executor.submit(_plot_count, df),
            executor.submit(_plot_pair, df),
        ]
        wait(futures)


"""
    Osoba B
"""
# Stałe
PARAM_GRID_LOGREG = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l1", "l2"],
}

PARAM_GRID_RF = {
    "clf__n_estimators": [200, 500],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_leaf": [1, 5, 10],
}

PARAM_GRID_XGB = {
    "clf__n_estimators": [200, 500],
    "clf__max_depth": [3, 6, 9],
    "clf__learning_rate": [0.05, 0.1],
    "clf__subsample": [0.8, 1.0],
}

PARAM_GRID_SVM = {
    "clf__C": [0.1, 1, 10],
    "clf__gamma": ["scale", 0.01, 0.1],
}

# Regresja liniowa, Random Forest, Gradient Boosting, SVM z jądrem RBF
def train_model(name, classifier, param_grid, preprocessor, X_train, y_train, model_path=None):
    if model_path is None:
        model_path = f"models/{name}.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", classifier),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    print(f"[{name}] Best params: {grid.best_params_}")
    print(f"[{name}] Best CV ROC-AUC: {grid.best_score_:.4f}")

    joblib.dump(grid.best_estimator_, model_path)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

# Wyciąga i zapisuje ważność cech dla modeli, które ją udostępniają (np. RF, XGB) lub współczynniki dla modeli liniowych (LogReg, SVM liniowy).
def compute_feature_importances(model, model_name, X_train, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    clf = model.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = abs(clf.coef_[0])  # dla LogReg/SVM liniowego
    else:
        print(f"[{model_name}] brak feature_importances_ ani coef_")
        return None

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}) \
               .sort_values("importance", ascending=False)
    imp_df.to_csv(f"{output_dir}/importance_{model_name}.csv", index=False)
    return imp_df

# Model hybrydowy
class HybridSoftVoter:
  
    def __init__(self, models: dict, weights: dict):
        self.models = models
        self.weights = weights
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        proba = np.zeros((len(X), 2))
        for name, model in self.models.items():
            proba += self.weights[name] * model.predict_proba(X)
        return proba

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
    
def hybryda(base_models: dict, cv_scores: dict, model_path: str = "models/hybrid.pkl"):

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    names = list(base_models.keys())
    raw = np.array([cv_scores[n] for n in names])
    weights = raw / raw.sum()
    weight_dict = dict(zip(names, weights))

    print(f"[hybrid] wagi: {weight_dict}")

    hybrid = HybridSoftVoter(base_models, weight_dict)
    joblib.dump(hybrid, model_path)

    # Zapis wag do raportu hiperparametry 
    pd.DataFrame(
        {"model": names, "cv_roc_auc": raw, "waga": weights}
    ).to_csv("tables/hybrid_wagi.csv", index=False)

    return hybrid

"""
    Osoba C
"""



if __name__ == "__main__":

    # Osoba A
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif"]

    X, Y = process_kepler_data("cumulative.csv")
    print(X.head())
    print(Y.head())

    save_descriptive_statistics(X)
    stats_table = pd.read_csv("tables/statystyki_opisowe.csv", index_col=0)
    pretty_print_descriptive_statistics(stats_table)

    df_viz = pd.concat([X, Y], axis=1)
    generate_visualizations(df_viz)

    # Osoba B
    """
    hp_results = []

    logreg_model, p, s = train_model(
    "logreg",
    LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
    PARAM_GRID_LOGREG, preprocessor, X_train, y_train)
    hp_results.append({"model": "logreg", "best_params": p, "cv_roc_auc": s})
    compute_feature_importances(logreg_model, "logreg", X_train)

    rf_model, p, s = train_model(
    "rf",
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
    PARAM_GRID_RF, preprocessor, X_train, y_train)
    hp_results.append({"model": "rf", "best_params": p, "cv_roc_auc": s})
    compute_feature_importances(rf_model, "rf", X_train)

    xgb_model, p, s = train_model(
    "xgboost",
    XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
    PARAM_GRID_XGB, preprocessor, X_train, y_train)
    hp_results.append({"model": "xgboost", "best_params": p, "cv_roc_auc": s})
    compute_feature_importances(xgb_model, "xgboost", X_train)

    svm_model, p, s = train_model(
    "svm_rbf",
    SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    PARAM_GRID_SVM, preprocessor, X_train, y_train)
    hp_results.append({"model": "svm_rbf", "best_params": p, "cv_roc_auc": s})
    compute_feature_importances(svm_model, "svm_rbf", X_train)

    os.makedirs("tables", exist_ok=True)
    pd.DataFrame(hp_results).to_csv("tables/hiperparametry.csv", index=False)

    base_models = {
    "logreg": logreg_model,
    "rf": rf_model,
    "xgboost": xgb_model,
    "svm_rbf": svm_model,
    }
    cv_scores = {row["model"]: row["cv_roc_auc"] for row in hp_results}

    hybrid_model = hybryda(base_models, cv_scores)   
    """