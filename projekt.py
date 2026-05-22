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
import missingno as msno
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn import set_config
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, confusion_matrix,
                             ConfusionMatrixDisplay)


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

# Analizuje braki danych i zwraca potok zawierający
# imputację KNN oraz winsoryzację.
def build_preprocessing_pipeline(X_train, requires_scaling=True):
    missing_ratio = X_train.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.4].index.tolist()

    active_cols = [c for c in X_train.columns if c not in cols_to_drop]

    skewed_candidates = ['okres_orbitalny', 'insolacja', 'glebokosc_tranzytu', 'promien_planety', 'stosunek_sygnal_szum']
    skewed_cols = [c for c in skewed_candidates if c in active_cols]
    other_cols = [c for c in active_cols if c not in skewed_cols]

    log_transformer = FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")

    col_transform = ColumnTransformer(transformers=[
        ('log1p', log_transformer, skewed_cols),
        ('passthrough', 'passthrough', other_cols)
    ], remainder='drop')

    scaler = StandardScaler() if requires_scaling else 'passthrough'

    pipeline = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('winsorizer', WinsorizerTransformer(0.01, 0.99)),
        ('transformations', col_transform),
        ('scaler', scaler)
    ])

    return pipeline, cols_to_drop


# Transformator do Pipeline wykonujący winsoryzację (1-99 percentyl).
# Uczy się granic na zbiorze treningowym i aplikuje je do testowego.
class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """
    Transformator do Pipeline wykonujący winsoryzację (1-99 percentyl).
    """
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
            self.feature_names_in_ = X_df.columns
        else:
            X_df = X
            self.feature_names_in_ = X.columns

        self.lower_bounds_ = X_df.quantile(self.lower_quantile)
        self.upper_bounds_ = X_df.quantile(self.upper_quantile)
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_df = X.copy()

        X_df = X_df.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        return X_df

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

"""
    Osoba B
"""
# Stałe
PARAM_GRID_LOGREG = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__l1_ratio": [1, 0],  
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


def evaluate_models_and_plot(models_dict, X_test, y_test, output_dir="figures"):
    # Osoba C - kod ewaluacji i wykresy do raportu
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    results = []
    
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        results.append({
            "model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc
        })
        
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FP", "CONF"])
        fig_cm, ax_cm = plt.subplots()
        disp.plot(ax=ax_cm, cmap="Blues")
        ax_cm.set_title(f"Macierz Pomyłek - {name}")
        fig_cm.savefig(f"{output_dir}/cm_{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_cm)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        
        pr, rc, _ = precision_recall_curve(y_test, y_proba)
        ax_pr.plot(rc, pr, label=f"{name} (AUC = {pr_auc:.3f})")
        
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Krzywe ROC")
    ax_roc.legend(loc="lower right")
    fig_roc.savefig(f"{output_dir}/09_roc_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig_roc)
    
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Krzywe Precision-Recall")
    ax_pr.legend(loc="lower left")
    fig_pr.savefig(f"{output_dir}/10_pr_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig_pr)
    
    res_df = pd.DataFrame(results)
    res_df.to_csv("tables/wyniki.csv", index=False)
    return res_df

def statistical_tests(models_dict, X_test, y_test):
    # Osoba C - kod testów statystycznych (McNemar)
    os.makedirs("results", exist_ok=True)
    names = list(models_dict.keys())
    preds = {name: models_dict[name].predict(X_test) for name in names}
    
    lines = ["Test McNemara na zbiorze testowym (p-values):\n"]
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name1, name2 = names[i], names[j]
            correct1 = (preds[name1] == y_test).astype(int)
            correct2 = (preds[name2] == y_test).astype(int)
            
            n11 = np.sum((correct1 == 1) & (correct2 == 1))
            n10 = np.sum((correct1 == 1) & (correct2 == 0))
            n01 = np.sum((correct1 == 0) & (correct2 == 1))
            n00 = np.sum((correct1 == 0) & (correct2 == 0))
            
            table = [[n11, n10], [n01, n00]]
            res = mcnemar(table, exact=False, correction=True)
            lines.append(f"{name1} vs {name2} -> p-value={res.pvalue:.4e}")
            
    with open("results/wilcoxon_test.txt", "w") as f:
        f.write("\n".join(lines))

def run_synthetic_demo(models_dict, X_template):
    # Osoba C - demo na sztucznych obserwacjach
    synths = []
    scenarios = ["Earth 2.0", "Hot Jupiter", "Eclipsing binary", "Noise", "Borderline"]
    for _ in scenarios:
        synths.append(X_template.iloc[0].copy())
        
    synths_df = pd.DataFrame(synths)
    synths_df.index = scenarios
    
    if "okres_orbitalny" in synths_df.columns:
      all_columns = [
          "okres_orbitalny", "czas_trwania", "glebokosc_tranzytu", "promien_planety",
          "temperatura_rownowagi", "insolacja", "stosunek_sygnal_szum",
          "temperatura_efektywna", "log_grawitacji", "promien_gwiazdy",
          "parametr_zderzenia", "jasnosc_kepler"
      ]

      # Earth 2.0 - przypomina Ziemię (gwiazda podobna do Słońca)
      synths_df.loc["Earth 2.0", all_columns] = [
          365.0, 13.0, 84.0, 1.0, 288.0, 1.0, 50.0, 5778.0, 4.44, 1.0, 0.1, 12.0
      ]

      # Hot Jupiter - gazowy gigant bardzo blisko gwiazdy
      synths_df.loc["Hot Jupiter", all_columns] = [
          3.0, 3.0, 10000.0, 11.0, 1500.0, 1000.0, 100.0, 5778.0, 4.44, 1.0, 0.5, 11.0
      ]

      # Eclipsing binary - zaćmieniowy układ podwójny (fałszywy alarm, ogromny spadek jasności)
      synths_df.loc["Eclipsing binary", all_columns] = [
          5.0, 5.0, 50000.0, 25.0, 1000.0, 500.0, 5.0, 6000.0, 4.20, 1.5, 0.9, 13.0
      ]

      # Noise - przypadkowy szum w danych słabej gwiazdy
      synths_df.loc["Noise", all_columns] = [
          10.0, 1.0, 10.0, 0.5, 500.0, 10.0, 4.0, 5000.0, 4.50, 0.8, 0.5, 16.0
      ]

      # Borderline - słaby sygnał, graniczny przypadek do weryfikacji
      synths_df.loc["Borderline", all_columns] = [
          50.0, 4.0, 300.0, 2.0, 400.0, 5.0, 10.0, 5500.0, 4.40, 0.9, 0.3, 14.0
      ]

    res = []
    for name, model in models_dict.items():
        proba = model.predict_proba(synths_df)[:, 1]
        res.append(pd.Series(proba, index=scenarios, name=name))
        
    demo_res = pd.concat(res, axis=1)
    demo_res.to_csv("tables/demo_syntetyki.csv")
    print("\nDemo syntetyków - Prawdopodobieństwo CONFIRMED:")
    print(demo_res.round(3))



if __name__ == "__main__":
    set_config(transform_output="pandas")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif"]

    # Osoba A
    X, y = process_kepler_data("cumulative.csv")
    print(X.head())
    print(y.head())

    save_descriptive_statistics(X)
    stats_table = pd.read_csv("tables/statystyki_opisowe.csv", index_col=0)
    pretty_print_descriptive_statistics(stats_table)

    df_viz = pd.concat([X, y], axis=1)
    generate_visualizations(df_viz)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    preprocessor_scaled, cols_to_drop_scaled = build_preprocessing_pipeline(X_train, requires_scaling=True)
    preprocessor_tree, cols_to_drop_tree = build_preprocessing_pipeline(X_train, requires_scaling=False)

    # Osoba B
    hp_results = []

    logreg_model, p, s = train_model(
    "logreg",
    LogisticRegression(solver="saga", max_iter=2000, random_state=42, class_weight="balanced"),
    PARAM_GRID_LOGREG, preprocessor_scaled, X_train, y_train)
    hp_results.append({"model": "logreg", "best_params": p, "cv_roc_auc": s})
    compute_feature_importances(logreg_model, "logreg", X_train)

    rf_model, p, s = train_model(
    "rf",
    RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced"),
    PARAM_GRID_RF, preprocessor_tree, X_train, y_train)
    hp_results.append({"model": "rf", "best_params": p, "cv_roc_auc": s})
    compute_feature_importances(rf_model, "rf", X_train)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model, p, s = train_model(
    "xgboost",
    XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1, scale_pos_weight=scale_pos_weight),
    PARAM_GRID_XGB, preprocessor_tree, X_train, y_train)
    hp_results.append({"model": "xgboost", "best_params": p, "cv_roc_auc": s})
    compute_feature_importances(xgb_model, "xgboost", X_train)

    svm_model, p, s = train_model(
    "svm_rbf",
    SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    PARAM_GRID_SVM, preprocessor_scaled, X_train, y_train)
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

    # Osoba C - wywołanie metod testowania i dema
    wszystkie_modele = base_models.copy()
    wszystkie_modele["hybrid"] = hybrid_model
    
    wyniki_df = evaluate_models_and_plot(wszystkie_modele, X_test, y_test)
    print("\nWyniki modeli na zbiorze testowym (Hold-out 20%):")
    print(wyniki_df.round(4).to_string(index=False))
    
    statistical_tests(wszystkie_modele, X_test, y_test)
    run_synthetic_demo(wszystkie_modele, X_test)
