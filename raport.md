**Znaleziska**
- **High:** `GridSearchCV` dla `LogisticRegression` używa `clf__l1_ratio` przy `solver="liblinear"`, co kończy się `ValueError` (l1_ratio działa tylko z `penalty="elasticnet"` i `solver="saga"`), więc trening logreg może się nie uruchomić. [projekt.py](projekt.py#L304) [projekt.py](projekt.py#L587)
- **Medium:** `compute_feature_importances` nie zapisze nic dla `SVC(kernel="rbf")`, bo model nie ma `coef_` ani `feature_importances_`, więc zabraknie ważności cech dla SVM w wynikach/raporcie. [projekt.py](projekt.py#L354-L372) [projekt.py](projekt.py#L612)
- **Medium:** `KNNImputer` działa na surowych skalach i zanim faktycznie odfiltrujesz kolumny z >40% braków (są tylko pomijane w `ColumnTransformer`), co może zniekształcić imputację i niepotrzebnie używać słabych cech w dystansie. [projekt.py](projekt.py#L235-L260)
- **Low:** Wyniki testu McNemara zapisujesz do pliku o nazwie sugerującej Wilcoxona, co wprowadza błąd semantyczny w raporcie. [projekt.py](projekt.py#L501)
- **Low:** Generowanie wykresów w `ProcessPoolExecutor` bywa kruche w środowiskach bez `fork`/GUI (multiprocessing + matplotlib); na Linux raczej ok, ale to potencjalny punkt awarii. [projekt.py](projekt.py#L222)

**Mocne strony**
- Usunięcie kolumn typu leakage/ID i selekcja klas `CONFIRMED` vs `FALSE POSITIVE` są zgodne z wymaganiami i redukują ryzyko przecieku informacji. [projekt.py](projekt.py#L49) [projekt.py](projekt.py#L62-L66)
- Preprocessing jest spięty w pipeline (imputacja, winsoryzacja, log1p, skalowanie), co ogranicza leakage i ułatwia replikowalność. [projekt.py](projekt.py#L235-L260)
- Walidacja obejmuje stratified CV i osobny hold-out oraz pełny zestaw metryk (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC). [projekt.py](projekt.py#L326-L345) [projekt.py](projekt.py#L416-L468) [projekt.py](projekt.py#L576-L578)
- Hybryda soft-voting z wagami z CV spełnia wymaganie „modelu hybrydowego” i jest czytelna do opisu w raporcie. [projekt.py](projekt.py#L374-L406)

**Ograniczenia przeglądu**
- Przegląd statyczny bez uruchomienia pipeline; brak weryfikacji runtime i wygenerowanych artefaktów.

**Rekomendacje**
1. Napraw strojenie `LogisticRegression`: albo usuń `l1_ratio` i stroić `penalty` z `liblinear`, albo przejdź na `solver="saga"` i dodaj `penalty="elasticnet"` + `l1_ratio`. [projekt.py](projekt.py#L304) [projekt.py](projekt.py#L587)
2. Dla SVM RBF użyj permutation importance (np. `sklearn.inspection.permutation_importance`) albo wyłącz SVM z sekcji „feature importance”. [projekt.py](projekt.py#L354-L372) [projekt.py](projekt.py#L612)
3. Zmień nazwę pliku z wynikami McNemara albo faktycznie wykonaj Wilcoxona, żeby nazewnictwo było spójne z treścią. [projekt.py](projekt.py#L501)
4. Rozważ prostszą imputację (median) lub odrzuć kolumny >40% braków przed KNN, by ograniczyć wpływ skali i „słabych” cech na odległości. [projekt.py](projekt.py#L235-L260)