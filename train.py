import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from dvclive import Live
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score

iris = load_iris()
X = iris.data
y = iris.target
with Live() as live:
    live.log_param("i*j*k", 900)

    for i in range(1, 11):
        for j in range(2, 11):
            for k in range(1, 11):
                clf = DecisionTreeClassifier(max_depth=i, min_samples_split=j, min_samples_leaf=k)
                clf.fit(X, y)
                live.log_metric('Precision', precision_score(y, clf.predict(X), average='micro'))
                live.log_metric('Recall', recall_score(y, clf.predict(X), average='micro'))
                live.log_sklearn_plot("confusion_matrix", y, clf.predict(X), name="cm.json")
                live.next_step()