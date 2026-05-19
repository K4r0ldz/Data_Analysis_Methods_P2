# Projekt 2 — Metody Analizy Danych: Klasyfikacja egzoplanet (Kepler KOI)

## Kontekst

Projekt zaliczeniowy z przedmiotu *Metody Analizy Danych*. Zadaniem jest wykonanie pełnego cyklu analizy klasyfikacyjnej zgodnie z rubryką z `projekt2_opis.pdf`: EDA, ≥3 modele + 1 model hybrydowy (formalnie min. 4 mechanizmy decyzyjne), walidacja, mierniki, przykład na sztucznych obserwacjach, sprawozdanie w stylu artykułu naukowego z bibliografią.

**Zbiór danych:** [cumulative.csv](cumulative.csv) — NASA Kepler Cumulative KOI (Kepler Objects of Interest), 9564 obserwacji × 50 zmiennych, 3 oryginalne klasy: `CONFIRMED` (2293), `CANDIDATE` (2248), `FALSE POSITIVE` (5023).

**Wybory ustalone z użytkownikiem:**
- Problem: **klasyfikacja binarna** — `CONFIRMED` vs `FALSE POSITIVE` (odrzucamy `CANDIDATE` jako klasę nierozstrzygniętą; zostaje 7316 obserwacji, klasy ~31/69%).
- Forma oddania: **`.py` (skrypty/moduły) + osobny PDF** ze sprawozdaniem.
- Praca **zespołowa — 3 osoby**: Karol Dziuba, Aleksander Grzegrzułka, Adam Grzywacz.
- Rubryka (`projekt2_opis.pdf`) dopuszcza grupy do 5 osób, a wymagania minimalne (≥3 metody + hybryda, ≥1 miernik, ≥1 sposób walidacji) są stałe niezależnie od liczebności zespołu — celujemy powyżej minimum (5 mechanizmów decyzyjnych, testy statystyczne, SHAP).

**KRYTYCZNA pułapka — data leakage.** W zbiorze są kolumny będące wynikiem istniejącego pipeline'u NASA, które zawierają informację o targecie:
- `koi_pdisposition` — wstępna klasyfikacja Keplera (kopia targetu),
- `koi_score` — score z modelu NASA (silnie skorelowany z odpowiedzią),
- `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec` — flagi false-positive ustawiane manualnie po analizie.

**Trzeba je usunąć przed treningiem.** Inaczej model osiągnie ~99% accuracy na tricku — i prowadzący to zauważy. To jest też świetny temat do dyskusji w sprawozdaniu (sekcja "Methodology — feature filtering"), który pokazuje świadomość metodologiczną.

Cel pracy: zbudować rzetelny klasyfikator rozpoznający potwierdzone egzoplanety **wyłącznie z fizycznych pomiarów tranzytu i parametrów gwiazdy**, bez korzystania ze score'ów NASA.

---

## Podział zadań — zespół 3-osobowy

Podział **pionowy wg faz**: każda osoba prowadzi jeden etap end-to-end — kod w `projekt.py`, wygenerowane wykresy i tabele oraz odpowiadającą sekcję sprawozdania. Role są bezimienne (**Osoba A / B / C**) — zespół przydziela się sam.

| Rola | Etap (kroki planu) | Funkcje w `projekt.py` | Wykresy / tabele | Sekcja raportu |
|---|---|---|---|---|
| **Osoba A** — EDA i preprocessing | Krok 1, Krok 2 | `krok_1_wczytaj_i_przefiltruj`, `krok_2_statystyki`, `krok_3_wizualizacja`, `krok_4_outliery`, `zbuduj_preprocessor` | `figures/01`–`08`, `tab1_statystyki_opisowe`, `tab2_outliery` | 6.1 Cel, 6.2 Wstępna analiza danych |
| **Osoba B** — Modele i hybryda | Krok 3, Krok 4 | `krok_6_modele`, `krok_7_hybryda`, `krok_9_importance` | `figures/11_importance_*`, `tab3_hiperparametry`, `importance_*.csv` | 7 Opis metod (5 metod + hybryda) |
| **Osoba C** — Walidacja, demo, raport | Krok 5, Krok 6 | `licz_metryki`, `krok_8_ewaluacja`, `krok_10_demo`, `main` | `figures/cm_*`, `09_roc_all`, `10_pr_all`, `tab4_wyniki`, `tab5_demo_syntetyki`, `wilcoxon_test.txt` | 8 Rezultaty, 9 Demo + składanie raportu |

**Balans obciążeń:** Osoba A ma najcięższy kod (cała EDA i wizualizacja), Osoba B najcięższy opis metod (5 metod z cytowaniami i wzorami), Osoba C koordynuje raport i pisze części wspólne (abstrakt, wstęp, wnioski). Obciążenia wyrównują się między kodem a tekstem.

**Punkt styku:** `feature importance` (`krok_9_importance`) należy do Osoby B, ale rysunki `11_importance_*` trafiają do sekcji 8 Rezultaty redagowanej przez Osobę C — uzgodnić podpisy i numerację rysunków.

Kroki **wspólne**: Krok 0 (środowisko, repo), Krok 7 (sprawozdanie — patrz tabela w Kroku 7), Krok 8 (checklist + weryfikacja końcowa).

---

## Krok 0 — Środowisko i struktura projektu (wspólne)

**Założyć venv** (`python -m venv .venv && source .venv/bin/activate`) i zainstalować:

| Biblioteka | Do czego |
|---|---|
| `pandas`, `numpy` | dane, statystyki |
| `scipy.stats` | skośność, kurtoza, testy statystyczne (Wilcoxon, McNemar) |
| `matplotlib`, `seaborn` | wykresy |
| `missingno` | wizualizacja braków danych |
| `scikit-learn` | pipeline, modele, walidacja, mierniki |
| `xgboost` lub `lightgbm` | gradient boosting |
| `imbalanced-learn` | (opcjonalnie) SMOTE / class_weight |
| `shap` | (opcjonalnie, na +) interpretacja modeli |
| `joblib` | zapis modeli do `.pkl` |

Zapisać `req.txt` (`pip freeze > req.txt`) — to liczy się jako profesjonalizm w sprawozdaniu (sekcja "Reproducibility").

**Sugerowana struktura plików:**

```
Projekt_2/
├── cumulative.csv
├── projekt.py                  # cały kod: EDA, preprocessing, modele, ewaluacja, demo
├── req.txt
├── README.MD
├── figures/                    # wszystkie wykresy do raportu (PNG, dpi=300)
├── results/                    # tabele z wynikami (CSV)
├── models/                     # zapisane .pkl
└── report/
    ├── sprawozdanie.tex        # (lub .docx)
    ├── references.bib
    └── sprawozdanie.pdf        # finalny PDF
```

**Organizacja kodu w `projekt.py`:** podzielić skrypt na sekcje oddzielone komentarzami (`# === Krok 1: EDA ===`, `# === Krok 2: Preprocessing ===` itd.) albo opakować każdy krok w funkcję (`def krok_1_eda():`, `def krok_2_preprocessing():` ...) wywoływaną w `if __name__ == "__main__":`. Pozwala to puścić cały pipeline jednym `python projekt.py` lub zaimportować i uruchamiać kroki pojedynczo.

**Reprodukowalność:** `np.random.seed(42)`, `random.seed(42)`, `random_state=42` w każdym modelu i splicie.

**Współpraca zespołowa nad `projekt.py`:** `projekt.py` to jeden plik, ale funkcje `krok_*` są rozłączne — każda osoba edytuje wyłącznie swoje funkcje (patrz „Podział zadań"), więc git scala zmiany per-funkcja bez konfliktów. Fragmenty wspólne (blok `import`, stałe `ID_COLS` / `LEAKAGE_COLS` / `SKY_COLS` / `LOG_FEATURES`, funkcja `main()`) koordynuje **Osoba C** — każda zmiana w nich zgłaszana na czacie zespołu. Sugerowany workflow: gałąź per osoba (`feature/osoba-a-eda`, `feature/osoba-b-modele`, `feature/osoba-c-ewaluacja`), przegląd przez pozostałe dwie osoby przed scaleniem do `main`.

---

## Krok 1 — Wstępna analiza danych (EDA) (Osoba A)

Skrypt: [projekt.py](projekt.py) — sekcja EDA

### 1.1 Wczytanie i kontekst zmiennych

- `pd.read_csv('cumulative.csv')`.
- Tabela ze znaczeniem kolumn (do raportu — sekcja "Variables description"):
  - `koi_disposition` — **target** (CONFIRMED / FALSE POSITIVE / CANDIDATE).
  - `koi_period` — okres orbitalny (dni).
  - `koi_duration` — czas trwania tranzytu (h).
  - `koi_depth` — głębokość tranzytu (ppm).
  - `koi_prad` — promień planety (R⊕).
  - `koi_teq` — temperatura równowagi (K).
  - `koi_insol` — insolacja (S⊕).
  - `koi_model_snr` — stosunek sygnał/szum.
  - `koi_steff` — temperatura efektywna gwiazdy (K).
  - `koi_slogg` — logarytm grawitacji powierzchniowej.
  - `koi_srad` — promień gwiazdy (R☉).
  - `koi_impact` — parametr zderzenia.
  - `koi_kepmag` — jasność w paśmie Keplera.
  - `ra`, `dec` — pozycja na niebie (raczej do odrzucenia jako pozbawiona sensu fizycznego dla klasyfikacji).
- W raporcie krótki opis fizyki tranzytu (1 akapit + cytowanie Borucki et al. 2010).

### 1.2 Filtrowanie kolumn (data leakage + identyfikatory)

**Bezwzględnie wyrzucić:**
- ID: `rowid`, `kepid`, `kepoi_name`, `kepler_name`, `koi_tce_delivname`.
- Leakage: `koi_pdisposition`, `koi_score`, `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec`.

**Do decyzji (uzasadnić w raporcie):**
- Kolumny błędów `_err1`/`_err2` — zostawić tylko sam pomiar (uproszczenie). Można wybrać alternatywę: zachować jako miarę niepewności, ale to podwaja liczbę cech.
- `ra`, `dec` — do odrzucenia (pozycja na niebie nie ma związku przyczynowego z klasą).

**Filtrowanie wierszy:** zachować tylko `koi_disposition ∈ {CONFIRMED, FALSE POSITIVE}`. Zakodować `y = 1` dla CONFIRMED, `y = 0` dla FALSE POSITIVE.

### 1.3 Statystyki opisowe

Wymagane minimum: średnia, mediana, min, max, odch. std., skośność. Polecane na +: kurtoza, kwartyle, IQR, % braków.

- `df.describe()` (mean, std, min, 25%, 50%, 75%, max).
- `scipy.stats.skew(df[col].dropna())`, `scipy.stats.kurtosis(...)`.
- Wszystko złączyć w jedną tabelę: `pd.concat([describe_df, skew_kurt_df])`, eksport do CSV i wstawić do raportu jako Tabelę 1.

### 1.4 Wizualizacja

Każdy wykres zapisać `plt.savefig('figures/...png', dpi=300, bbox_inches='tight')`.

- **Histogram + KDE** dla każdej zmiennej numerycznej (`sns.histplot(..., kde=True)`) — zauważyć silną prawoskośność `koi_period`, `koi_insol`, `koi_depth`, `koi_prad`.
- **Boxplot** dla każdej zmiennej z podziałem na klasę (`sns.boxplot(x='target', y=col)`) — pokazuje zarówno outliery, jak i separowalność klas.
- **Macierz korelacji** (`sns.heatmap(df.corr(), cmap='coolwarm', center=0)`) — pokazać multikolinearność (np. `koi_teq` ↔ `koi_insol`).
- **Countplot targetu** (`sns.countplot(x='target')`) — pokazać balans klas.
- **Pairplot** dla 4-5 najważniejszych zmiennych z `hue='target'` — wizualnie pokazać separowalność.
- **ECDF** lub **violin plot** jako wartość dodana.

### 1.5 Braki danych

- `df.isna().sum().sort_values(ascending=False)` + `missingno.matrix(df)` lub `missingno.bar(df)`.
- Strategie obsługi (do opisu w raporcie + decyzja):
  - Kolumny z >40% braków → drop (raczej `koi_teq_err1/2`).
  - Kolumny z <40% braków → imputacja **medianą** (odporna na skośność) przez `sklearn.impute.SimpleImputer(strategy='median')` lub lepiej `KNNImputer(n_neighbors=5)` / `IterativeImputer` (na +).
- **Imputację robić wewnątrz `Pipeline`**, żeby nie wyciekała informacja z testu do treningu (klasyczny błąd).

### 1.6 Obserwacje odstające

- Detekcja: reguła IQR (Q1 − 1.5·IQR, Q3 + 1.5·IQR), Z-score (`scipy.stats.zscore`), `sklearn.ensemble.IsolationForest`, `sklearn.neighbors.LocalOutlierFactor`.
- Obsługa (wybrać i uzasadnić):
  - **Winsoryzacja** (`scipy.stats.mstats.winsorize`) na 1-99 percentylu — preferowane, nie tracimy obserwacji.
  - Albo: usunięcie skrajnych wierszy (ostrożnie, można wyciąć rzadkie ciekawe planety).
  - Albo: nie usuwać + użyć modeli odpornych (drzewa).
- W raporcie: tabela "n outlierów per kolumna" + komentarz, że dla drzew nie usuwamy, a dla regresji logistycznej i SVM stosujemy `RobustScaler`.

### 1.7 Transformacje

- **Logarytmowanie** silnie skośnych: `np.log1p` na `koi_period`, `koi_insol`, `koi_depth`, `koi_prad`, `koi_model_snr`. Pokazać przed/po (histogramy + skośność) — to sztandarowy materiał na raport.
- **Skalowanie:** `StandardScaler` dla regresji logistycznej / SVM / sieci, `RobustScaler` jeśli zostawiamy outliery. Drzewa (RF, XGBoost) nie wymagają skalowania.
- Wszystko zapakować w `sklearn.pipeline.Pipeline` + `ColumnTransformer` — jeden obiekt pipeline'u na model.

---

## Krok 2 — Podział danych (Osoba A)

Skrypt: [projekt.py](projekt.py) — sekcja preprocessing

- `from sklearn.model_selection import train_test_split, StratifiedKFold`.
- Strategia: **trening 80% / test 20%** (`stratify=y`, `random_state=42`) jako *hold-out test set* — używany wyłącznie raz na końcu do raportowania ostatecznych wyników.
- Na zbiorze treningowym: **`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`** do tuningu hiperparametrów i porównania modeli.
- Bonus na +: `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` — bardziej wiarygodne estymaty.

---

## Krok 3 — Modele bazowe (3 metody, ale zaplanuj 4) (Osoba B)

Skrypt: [projekt.py](projekt.py) — sekcja modeli

Wymagane minimum: **3 metody + 1 hybryda = 4 mechanizmy**. Sugeruję zrobić **4 metody + 1 hybryda = 5 mechanizmów** dla pełnych punktów.

Każda metoda w raporcie: krótka geneza, intuicja, wzór/algorytm w 3-5 zdaniach, **cytowanie pierwotnej pracy**.

| # | Metoda | Klasa sklearn | Cytowanie |
|---|---|---|---|
| 1 | Regresja logistyczna | `LogisticRegression` | Cox (1958), Hosmer & Lemeshow (2000) |
| 2 | Random Forest | `RandomForestClassifier` | Breiman (2001) |
| 3 | Gradient Boosting | `XGBClassifier` (xgboost) lub `LGBMClassifier` | Chen & Guestrin (2016) / Ke et al. (2017) |
| 4 | SVM (RBF) | `SVC(kernel='rbf', probability=True)` | Cortes & Vapnik (1995) |
| (5) | k-NN albo MLP | `KNeighborsClassifier` / `MLPClassifier` | Cover & Hart (1967) / Rumelhart et al. (1986) |

**Tuning:** `GridSearchCV` lub `RandomizedSearchCV` z `cv=StratifiedKFold(5)`, `scoring='roc_auc'`. Zakresy hiperparametrów (przykład):
- LogReg: `C ∈ {0.01, 0.1, 1, 10}`, `penalty ∈ {'l1', 'l2'}`, `solver='liblinear'`.
- RF: `n_estimators ∈ {200, 500}`, `max_depth ∈ {None, 10, 20}`, `min_samples_leaf ∈ {1, 5, 10}`.
- XGBoost: `n_estimators ∈ {200, 500}`, `max_depth ∈ {3, 6, 9}`, `learning_rate ∈ {0.05, 0.1}`, `subsample ∈ {0.8, 1.0}`.
- SVM: `C ∈ {0.1, 1, 10}`, `gamma ∈ {'scale', 0.01, 0.1}`.

Zapisać w tabeli wybrane hiperparametry — wstawić do raportu.

---

## Krok 4 — Model hybrydowy (mechanizm #4) (Osoba B)

Wymóg z opisu: *„np. średnia ważona score'ów z metod"*. Trzy ścieżki:

1. **Najprostsza i zgodna z opisem (rekomendowana):** soft voting z wagami = AUC z CV danego modelu, znormalizowane do sumy 1. Implementacja samodzielna: `proba_hybrid = sum(w_i * model_i.predict_proba(X)[:,1])`. Lub `sklearn.ensemble.VotingClassifier(estimators=..., voting='soft', weights=[...])`.
2. **Stacking** (na +): `sklearn.ensemble.StackingClassifier` z `LogisticRegression` jako meta-uczniem na predykcjach modeli bazowych. Wymaga uważnego CV wewnątrz, żeby nie wyciekły dane.
3. **Bayesian Model Averaging** — przesada na ten projekt, wspomnieć w "Future work".

W raporcie napisać explicite, że hybryda to **czwarty mechanizm decyzyjny** spełniający wymóg z opisu projektu.

---

## Krok 5 — Walidacja i mierniki (Osoba C)

Skrypt: [projekt.py](projekt.py) — sekcja ewaluacji

### 5.1 Mierniki (wybrać kilka, nie tylko jeden)

Z `sklearn.metrics`:
- **Accuracy** (`accuracy_score`).
- **Precision, Recall, F1** (`precision_score`, `recall_score`, `f1_score`, `classification_report`).
- **ROC-AUC** (`roc_auc_score`) + krzywe ROC (`roc_curve` + `RocCurveDisplay`).
- **PR-AUC / average precision** (`average_precision_score`, `PrecisionRecallDisplay`) — istotne, bo klasy lekko niezbalansowane.
- **Macierz pomyłek** (`confusion_matrix`, `ConfusionMatrixDisplay`).
- **Log-loss** (`log_loss`) — porównanie kalibracji.
- (Bonus) **Krzywe kalibracji** (`CalibrationDisplay`) — szczególnie ciekawe dla SVM vs RF.

### 5.2 Walidacja

- **5-fold StratifiedKFold** na zbiorze treningowym (raportować średnią ± std dla każdego miernika).
- **Hold-out 20%** test na końcu — finalna tabela porównawcza w raporcie.
- (Bonus) **Powtórzona K-Fold** (`RepeatedStratifiedKFold(n_repeats=3)`) — odporniejsza estymata.
- (Bonus) **Test statystyczny** porównujący modele:
  - **Wilcoxon signed-rank test** na fold-wise AUC (`scipy.stats.wilcoxon`) — czy różnica między modelami istotna.
  - lub **McNemar** na predykcjach holdout (`statsmodels.stats.contingency_tables.mcnemar`).

### 5.3 Wyniki — wykresy do raportu

- Tabela: model × miernik (mean ± std z CV + wartość na hold-out).
- Krzywe ROC dla 5 mechanizmów na jednym wykresie (pomocne do dyskusji).
- Krzywe PR analogicznie.
- Macierze pomyłek (4-5 paneli, jeden na model).
- Bar plot: feature importance dla RF + SHAP summary plot dla XGBoost (na +).

---

## Krok 6 — Przykład użycia na sztucznych obserwacjach (Osoba C)

Skrypt: [projekt.py](projekt.py) — sekcja demo na sztucznych obserwacjach

Stworzyć ręcznie **3-5 fikcyjnych KOI** jako `pd.DataFrame` z fizycznie sensownymi wartościami i opisem scenariusza. Przykłady:

| Scenariusz | Oczekiwana klasa | Charakterystyka |
|---|---|---|
| "Earth 2.0" | CONFIRMED | `koi_period≈365`, `koi_prad≈1.0`, `koi_teq≈288`, `koi_depth≈84`, wysoki SNR |
| "Hot Jupiter" | CONFIRMED | `koi_period≈3`, `koi_prad≈11`, `koi_teq≈1500`, `koi_depth≈10000`, wysoki SNR |
| "Eclipsing binary" | FALSE POSITIVE | `koi_depth≈50000`, `koi_prad>20`, `koi_duration` długi, niski SNR |
| "Noise" | FALSE POSITIVE | `koi_model_snr<8`, parametry przypadkowe |
| "Borderline" | niepewne | wartości średnie — dyskusja, jak modele różnią się |

Dla każdego: przepuścić przez ten sam pipeline (`pipeline.predict(X_synth)`, `pipeline.predict_proba(X_synth)`), pokazać predykcję każdego z 5 mechanizmów + interpretację. To świetna wisienka na torcie raportu.

---

## Krok 7 — Sprawozdanie PDF (artykuł naukowy) (wspólne)

Folder: [report/](report/)

**Rekomendacja narzędziowa:** LaTeX (Overleaf) z gotowym szablonem. W opisie projektu jest link do szablonu MIBE (`MIBE_Szablon2019.docx`) — można go pobrać i przerobić, albo użyć dowolnego szablonu IEEE/Elsevier. Obrazy `.png` z folderu `figures/` wstawiać przez `\includegraphics`. Bibliografia w BibTeX (`references.bib`), styl `apalike` lub `IEEEtran`.

### Struktura (1:1 z rubryką w `projekt2_opis.pdf`):

1. **Tytuł** — np. *"Klasyfikacja kandydatów na egzoplanety z misji Kepler z wykorzystaniem modeli uczenia maszynowego"*.
2. **Autorzy** — Karol Dziuba, Aleksander Grzegrzułka, Adam Grzywacz (afiliacja, e-maile).
3. **Streszczenie (≤150 słów)** — problem + dane + metody + wynik (jedno zdanie na każde). Liczyć słowa!
4. **Słowa kluczowe** — 5-7, np.: klasyfikacja, uczenie maszynowe, egzoplanety, Kepler, Random Forest, XGBoost, model hybrydowy.
5. **Wprowadzenie** — kontekst astronomiczny (tranzyty, misja Kepler) + motywacja ML + cel pracy + struktura artykułu.
6. **Przedmiot badania:**
   - 6.1 Cel.
   - 6.2 Wstępna analiza danych:
     - opis zmiennych (Tabela);
     - statystyki opisowe (Tabela: średnia, mediana, min, max, std, skośność, [kurtoza]);
     - wizualizacja (boxplot, histogramy, korelacja, countplot targetu);
     - transformacje (log1p, skalowanie — przed/po);
     - braki danych (% per kolumna + strategia imputacji);
     - obserwacje odstające (metoda detekcji + sposób obsługi);
     - **decyzje o filtrowaniu kolumn (data leakage)** — kluczowy akapit!
7. **Opis metod** — dla każdej z 5 metod (4 bazowe + hybryda) podsekcja:
   - intuicja, mechanizm, wzór jeśli zwięzły;
   - cytowanie pierwotnej pracy;
   - hiperparametry wybrane przez grid search.
8. **Rezultaty:**
   - sposób walidacji (5-fold stratified + hold-out 20%);
   - tabela: model × {Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC} (mean ± std);
   - krzywe ROC (jeden wykres);
   - macierze pomyłek;
   - feature importance / SHAP (na +);
   - test statystyczny porównawczy (na +).
9. **Przykład użycia na sztucznych obserwacjach** — Tabela ze scenariuszami + tabela predykcji każdego modelu + krótka dyskusja, gdzie modele się rozjeżdżają.
10. **Wnioski / Dyskusja** — co zadziałało, co nie, ograniczenia (np. odrzucenie CANDIDATE), kierunki dalsze (sieci 1D-CNN na krzywych światła, kalibracja).
11. **Bibliografia** — patrz niżej.

### Podział sekcji raportu między autorów

| Sekcja raportu | Autor |
|---|---|
| 1 Tytuł, 2 Autorzy | wspólne (składa Osoba C) |
| 3 Streszczenie, 4 Słowa kluczowe, 5 Wprowadzenie | Osoba C |
| 6.1 Cel, 6.2 Wstępna analiza danych | Osoba A |
| 7 Opis metod (5 metod + hybryda) | Osoba B |
| 8 Rezultaty | Osoba C |
| 9 Przykład użycia na sztucznych obserwacjach | Osoba C |
| 10 Wnioski / Dyskusja | wspólne — każdy dorzuca akapit o swojej części, redaguje Osoba C |
| 11 Bibliografia | wspólne — każdy dodaje cytowania swojej części do `references.bib`, scala Osoba C |

**Składanie LaTeX/Overleaf** i wygenerowanie finalnego PDF — Osoba C. Wszystkie osoby pracują w jednym projekcie Overleaf (kontrola wersji online eliminuje konflikty scalania).

### Bibliografia — minimalny zestaw

```
Borucki, W. J., et al. (2010). Kepler planet-detection mission: introduction and first results. Science, 327(5968), 977-980.
Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD '16, 785-794.
Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.
Cox, D. R. (1958). The regression analysis of binary sequences. JRSS Series B, 20(2), 215-242.
Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression (3rd ed.). Wiley.
McCauliff, S. D., et al. (2015). Automatic classification of Kepler planetary-transit candidates. ApJ, 806(1), 6.
Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.
Thompson, S. E., et al. (2018). Planetary candidates observed by Kepler. VIII. ApJS, 235(2), 38.
Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS.
```

(Dla każdego cytowania w tekście użyj `\citep{...}` w LaTeX.)

---

## Krok 8 — Checklist „pełne punkty" (przed oddaniem) (wspólne)

Sprawdź każdy punkt z opisu projektu literalnie. **Każda osoba odhacza pozycje ze swojego etapu**, a finalną weryfikację wykonuje cały zespół:

- [ ] Tytuł, autor, streszczenie ≤150 słów (policzyć!), słowa kluczowe, wprowadzenie.
- [ ] Cel jasno sformułowany w sekcji "Przedmiot badania".
- [ ] Wszystkie zmienne opisane (tabela).
- [ ] Statystyki: średnia, mediana, min, max, odch. std., **skośność** (często pomijana — sprawdzić!).
- [ ] Wizualizacja: boxplot **i** histogramy.
- [ ] Transformacje: skalowanie **i** logarytmowanie pokazane (przed/po).
- [ ] Braki danych: omówione + strategia obsłużona.
- [ ] Outliery: metoda + obsługa opisana.
- [ ] **≥3 metody + hybryda = ≥4 mechanizmy decyzyjne** (zaplanowano 5).
- [ ] Każda metoda **z cytowaniem** pierwotnej pracy.
- [ ] **≥1 miernik** (dam 6: Acc, Prec, Rec, F1, ROC-AUC, PR-AUC).
- [ ] **≥1 sposób walidacji** (dam 2: 5-fold CV + hold-out).
- [ ] **Przykład użycia na sztucznych obserwacjach** (tabela 3-5 scenariuszy).
- [ ] Bibliografia (≥10 pozycji).
- [ ] Format ≈ artykuł naukowy (sekcje, abstrakt, podpisy rys./tabel, numerowanie).
- [ ] Pliki do oddania: `cumulative.csv`, `sprawozdanie.pdf`, `projekt.py`, `req.txt`.
- [ ] Załadowanie na MS Teams **bez archiwizowania** (wymóg z opisu — luzem, nie .zip).
- [ ] **Wspólne czytanie całego raportu na głos** przez wszystkie 3 osoby — spójność stylu, terminologii i numeracji rysunków/tabel między sekcjami różnych autorów.

### Co wynosi pracę powyżej minimum (na realne 5.0):

- Świadome odrzucenie kolumn powodujących data leakage (sekcja w raporcie).
- Pipeline `sklearn.Pipeline` + `ColumnTransformer` (czysta, branżowa praktyka).
- 5 mechanizmów decyzyjnych zamiast 4.
- Test statystyczny porównujący modele (Wilcoxon / McNemar).
- Krzywe ROC i PR na wspólnych wykresach.
- Feature importance + SHAP.
- Reprodukowalność: `random_state=42` wszędzie, `req.txt`, `joblib.dump` modeli.
- Wykresy w 300 dpi, jednolita paleta, opisane osie i legendy.
- Polski tekst poprawny stylistycznie, bez literówek (przeczytać na głos przed oddaniem).

---

## Weryfikacja końcowa (jak sprawdzić, że projekt działa)

1. **Sanity test danych:** po filtrowaniu liczyć `df.shape` — powinno być ~7316 wierszy × ~30-40 kolumn (po odrzuceniu leakage/ID).
2. **Sanity test modelu:** dla regresji logistycznej z 3 cechami (`koi_depth`, `koi_prad`, `koi_model_snr`) ROC-AUC powinno być w okolicach 0.85-0.92. Dla pełnego XGBoost — 0.93-0.97. **Jeśli widzisz 0.99+ to znaczy, że nie usunąłeś leakage'u** — wróć do kroku 1.2.
3. **Sanity test syntetyków:** "Earth 2.0" powinno dać `predict_proba(CONFIRMED) > 0.5` w przynajmniej 3 z 5 modeli.
4. **Spróbuj uruchomić `projekt.py` od zera** w czystym venv z `req.txt` — czy wszystko działa.
5. Wyrenderuj PDF, otwórz, sprawdź wszystkie odnośniki do figur (`Fig. ??` = błąd kompilacji LaTeX).
6. Policzyć słowa w abstrakcie (`echo "..." | wc -w`) — musi być ≤150.
