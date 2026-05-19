import math
import os
from concurrent.futures import ProcessPoolExecutor, wait

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns


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


if __name__ == "__main__":
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
