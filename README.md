# AI-Machine-Learning

Practical work 2 for *Inteligência Artificial* (CC2006). It builds and evaluates classifiers on the experimental speed-dating data set (`speedDating_trab.csv`) to predict whether a date ends in a match, using two algorithms:

- Decision Tree (CART, scikit-learn)
- Gaussian Naive Bayes

Each is evaluated with hold-out and 5-fold cross-validation (accuracy, precision, recall, F1, confusion matrix).

## Layout

- `src/AI.py` – data cleaning (drop columns/rows, fill missing values with the mean), class balancing, training and evaluation
- `speedDating_trab.csv` – data set
- `relatorio.pdf` / `relatorio.odt` – report (Portuguese); `IA2021_TrabalhoPratico2_v2.pdf` – assignment
- `pictures/` – confusion matrices and trees from a run; `pictures_examples/` – the ones used in the report
- `anaconda/` – original conda environment (`environment.yml`, `spec-file.txt`)

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # also install Graphviz (e.g. brew install graphviz)
python src/AI.py
```

Scores are printed to the terminal. Plots are written to `pictures/` (this overwrites the committed ones).
