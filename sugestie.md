# Ocena raportu `Projekt_2_Lab1_GR_4.pdf` — rzeczy do poprawy

## Context
Zadanie: ocenić raport `Projekt_2_Lab1_GR_4.pdf` (19 s.) względem wytycznych z `projekt2_opis.pdf`
oraz względem kodu `projekt.py` i wygenerowanych artefaktów (`tables/*.csv`, `results/`, `figures/`).
Cel: wypisać wszystkie rzeczy do poprawy. To zadanie recenzyjne — poniżej jest lista
poprawek; ewentualne wprowadzenie ich (edycja tekstu, regeneracja rysunków) wymaga osobnej zgody.

## Zgodność z wytycznymi `projekt2_opis.pdf` — WSZYSTKIE wymagania formalne spełnione
- Zbiór: 12 zmiennych (≥10 ✓), 7316 obs. (≥800 ✓). Grupa 3 os. ✓
- Tytuł ✓, autorzy ✓, streszczenie ~120 słów (≤150 ✓), słowa kluczowe ✓, wprowadzenie ✓
- Cel ✓; wstępna analiza: zmienne ✓, statystyki opisowe (mean/mediana=50%/min/max/std/skośność + extra) ✓,
  wizualizacja (histogramy/boxplot/violin) ✓, transformacje (log1p + StandardScaler) ✓,
  braki danych ✓, obserwacje odstające (winsoryzacja) ✓
- Metody: 4 modele bazowe + hybryda = 5 mechanizmów (≥4 ✓), każdy z cytowaniem [10]–[15] i opisem ✓
- Rezultaty: wiele mierników ✓; walidacja: 5-fold CV + hold-out + McNemar (≥1 ✓)
- Sztuczne obserwacje ✓; bibliografia ✓
**Wniosek: raport jest kompletny i zgodny z wymaganiami. Poprawki dotyczą spójności, aktualności
artefaktów, dokładności liczb i języka — nie braków formalnych.**

## Weryfikacja (sklearn 1.8.0) — to NIE jest błąd
`penalty` jest `deprecated`; `l1_ratio` steruje regularyzacją (l1_ratio=1→L1/rzadkie, 0→L2). Grid
`clf__l1_ratio:[0,1]` działa poprawnie. Stara notatka `raport.md` (liblinear/ValueError) jest nieaktualna.

---

## KRYTYCZNE — niespójność raportu z kodem/danymi (różne przebiegi sklejone w raport)

1. **Nieaktualny rysunek braków danych (Rys. 3.5.3).** W PDF macierz ma **14 kolumn** z `koi_time0bk`
   i `koi_tce_plnt_num` — czyli zmiennymi, które tekst 3.3.1 deklaruje jako usunięte, a kod
   (`process_kepler_data`, projekt.py:53) faktycznie usuwa. Aktualny `figures/missing_matrix.png`
   ma poprawne **12 kolumn**. Rysunek w PDF zaprzecza własnemu tekstowi → podmienić na aktualny.

2. **Zła tabela pod sekcją 5.2.** Tekst 5.2 cytuje wartości z **hold-out (test)** (XGBoost Acc 0,931;
   logreg precision 0,709 — zgodne z `tables/wyniki.csv`), ale wyświetlona tabela to **5-fold CV**
   (`tables/wyniki_cv.csv`: 0,927±0,003 itd.) i **nie zawiera hybrydy**. Liczby w tabeli przeczą prozie,
   a kluczowy model (hybryda) znika z tabeli metryk.
   - Fix: pod 5.1/5.2 wstawić tabelę TEST (`wyniki.csv`, z hybrydą); tabelę CV ±std przenieść do 5.3
     (sekcja 5.3 nie ma teraz żadnej tabeli).

3. **Błędne p-value McNemara (5.4).** Raport: XGBoost vs Random Forest p=0,037. Aktualny
   `results/mcnemar_test.txt`: rf vs xgboost = **0,0280** (rf vs hybrid = 0,0350). 0,037 nie pasuje do
   żadnej bieżącej wartości → zsynchronizować z aktualnym wynikiem. (Wartość xgboost vs hybrid 0,710 ✓.)

4. **Nieaktualna tabela demo syntetycznego (Rys. 6.1).** Część wartości ≠ aktualne
   `tables/demo_syntetyki.csv`: Earth 2.0 RF 0,37 vs 0,35, hybryda 0,6635 vs 0,6584; Hot Jupiter RF
   0,346 vs 0,36, hybryda 0,1931 vs 0,1966; Borderline RF 0,08 vs 0,082. → regenerować.

> **Przyczyna źródłowa #1–#4:** raport miesza artefakty z różnych przebiegów kodu. Rekomendacja:
> uruchomić `projekt.py` raz na końcu i wstawić WSZYSTKIE rysunki/tabele z tego jednego przebiegu.

## ŚREDNIE — dokładność liczb i luki w treści

5. **Ucinanie zamiast zaokrąglania w prozie 5.1 (rozjazd z rysunkami):** XGBoost ROC-AUC „0,974" vs
   0,975 (0,9748); hybryda PR-AUC „0,950" vs 0,951; XGBoost PR-AUC „0,945" vs 0,946; logreg ROC-AUC
   „0,927" vs 0,928; SVM recall „0,938" vs 0,939; SVM precision „0,825" vs 0,826; hybryda F1 „0,887"
   vs 0,888. → zaokrąglać spójnie i zgodnie z rysunkami.

6. **Ważność cech obiecana, ale niepokazana.** Sekcje 3.2 i 4.2 podkreślają interpretowalność /
   `feature_importances_` jako główny argument za wyborem modeli, kod liczy i zapisuje
   `results/importance_*.csv` dla każdego modelu — a sekcja Rezultaty nie pokazuje żadnej ważności cech.
   → dodać tabelę/wykres ważności (dane już istnieją) albo złagodzić deklarację.

7. **Brak wybranych hiperparametrów.** Raport mówi o strojeniu GridSearchCV, `tables/hiperparametry.csv`
   istnieje, ale wybrane wartości nie są nigdzie pokazane. → rozważyć małą tabelę best-params (reprodukcja).

8. **Wątpliwa interpretacja „Hot Jupiter" (sekcja 6).** Hot Jupiter to realna (potwierdzona) klasa planet,
   a raport przedstawia jego odrzucenie przez modele jako sukces. → jawnie podać zakładaną prawdziwą klasę
   każdego z 5 profili (ground truth), żeby dało się ocenić poprawność, lub przeformułować narrację.

9. **Interpretacja McNemara nieprecyzyjna.** „test wykazał, że regresja radzi sobie gorzej" — McNemar
   bada istotność różnicy rozkładów błędów, nie kierunek; kierunek odczytuje się z metryk. Doprecyzować;
   podać konkretne p-value zamiast „rzędu 10⁻¹⁵ do 10⁻²⁰" (i poprawić formatowanie wykładników).

## NISKIE — język i formatowanie

10. **4.1 — zdublowane zdanie:** „...naturalny punkt odniesienia: jego skuteczność: jego skuteczność
    wyznacza poziom..." → usunąć duplikat.
11. **4.2 — literówki:** „co deklaruje pojedyncze drzewa i redukcje wariancje modelu" → „co **dekoreluje**
    pojedyncze drzewa i **redukuje wariancję** modelu".
12. **Streszczenie — brak orzeczenia:** „wagi modeli składowych proporcjonalne do ich jakości" → „...**są**
    proporcjonalne...".
13. **Nagłówek 4.3 „Gradient Boosting"** vs etykieta „xgboost" w tabelach/streszczeniu → ujednolicić, np.
    „Gradient Boosting (XGBoost)".
14. **Uboga bibliografia jak na styl artykułu naukowego:** pozycje tylko autor/rok/tytuł, [1] to surowy
    URL, brak czasopism/wydawców/DOI. → uzupełnić do spójnego stylu cytowań.
15. **Niespójny separator dziesiętny:** proza po polsku (0,974), tabele/rysunki z kodu po angielsku (0.974).
    → ujednolicić lub zaznaczyć, że rysunki są generowane programowo.
16. **Lista usuniętych zmiennych (3.3.1) pomija `koi_tce_plnt_num`** — kod ją usuwa jako wyciek
    („multiplicity boost"); to jeden z ciekawszych przypadków leakage → dopisać.
17. **Notacja „p=" dla prawdopodobieństw (sekcja 6):** „...jako egzoplanetę ... p=0,829", „p=0,004" myli
    się z p-value z 5.4 → użyć „prawdopodobieństwo = ...".

## Drobne obserwacje (opcjonalne)
- Wagi hybrydy są niemal równe (logreg 0,242 vs xgboost 0,253; `tables/hybrid_wagi.csv`), więc „premiowanie
  silniejszych modeli" jest praktycznie nieodróżnialne od prostej średniej — można o tym wspomnieć.
- Streszczenie wymienia tylko CV, pomija hold-out 20% (ciało raportu to pokrywa).

## Weryfikacja poprawek (gdyby je wprowadzać)
1. Uruchomić `.venv/bin/python projekt.py` (jeden przebieg) → świeże `tables/`, `figures/`, `results/`.
2. Podmienić w raporcie: Rys. 3.5.3 (12 kolumn), Rys. 6.1, tabele 5.x, p-value McNemara z `mcnemar_test.txt`.
3. Sprawdzić, że każda liczba w prozie 5.1–5.4 i sek. 6 zgadza się co do zaokrąglenia z rysunkiem/CSV.
4. Korekta językowa pkt 10–17.
