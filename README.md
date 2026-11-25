# AI-Machine-Learning

Practical work 2 for *Inteligência Artificial* (CC2006). It builds and evaluates classifiers on the experimental speed-dating data set (`speedDating_trab.csv`) to predict whether a date ends in a match, using two algorithms:

- Decision Tree (CART, scikit-learn)
- Gaussian Naive Bayes

Each is evaluated with hold-out and 5-fold cross-validation (accuracy, precision, recall, F1, confusion matrix).

## Layout

- `src/AI.py` – data cleaning (drop columns/rows, fill missing values with the mean), class balancing, training and evaluation
- `speedDating_trab.csv` – data set
- `report/relatorio.tex` / `report/relatorio.pdf` – revised report in LaTeX (Portuguese); `relatorio.pdf` / `relatorio.odt` – original 2021 report; `IA2021_TrabalhoPratico2_v2.pdf` – assignment
- `pictures/` – confusion matrices and trees from a run; `pictures_examples/` – the ones used in the report
- `anaconda/` – original conda environment (`environment.yml`, `spec-file.txt`)

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # also install Graphviz (e.g. brew install graphviz)
python src/AI.py
```

Build the report (needs a TeX distribution):

```bash
cd report && latexmk -pdf relatorio.tex
```

## Results (positive class: match)

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| CART, hold-out (70/30) | 0.764 | 0.303 | 0.333 | 0.317 |
| CART, 5-fold CV | 0.765 | 0.307 | 0.341 | 0.323 |
| Naive Bayes, hold-out (70/30) | 0.822 | 0.422 | 0.222 | 0.291 |
| Naive Bayes, 5-fold CV | 0.829 | 0.465 | 0.251 | 0.326 |

Stratified splits, missing values imputed with the training mean. The script also prints extra experiments (depth limit, class weights, without id/partner); the best F1 (0.41) is CART with depth 5 and balanced class weights, at the cost of accuracy (0.63).

Only ~16% of the examples are matches, so accuracy alone is misleading. See the report for limitations.

Scores are printed to the terminal. Plots are written to `pictures/` (this overwrites the committed ones).
