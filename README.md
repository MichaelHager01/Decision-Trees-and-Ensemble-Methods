# Decision trees and ensemble methods

Trains a decision tree on the moons dataset, then builds a 1,000-tree ensemble by majority vote.

## What it does

- Generates 10,000 moons samples with noise 0.4.
- Tunes a `DecisionTreeClassifier` with grid search and records test accuracy.
- Draws 1,000 random subsets of 100 training rows with `ShuffleSplit`, trains one tree on each, and combines them with a majority vote.

## Layout

- `notebooks/decision_trees.ipynb` — the tree, the forest, and the accuracy comparison

## Run

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/decision_trees.ipynb
```

## Stack

Python, scikit-learn, SciPy, Jupyter. Written for MEEN 423.
