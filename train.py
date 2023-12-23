import seaborn as sns
import matplotlib.pyplot as plt
from dvclive import Live
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix

with Live() as live:
    live.log_param("epochs", 1)

    for i in range(1, 5):
        for j in range(2, 5):
            for k in range(1, 5):
                plt.clf()
                clf = DecisionTreeClassifier(max_depth=i, min_samples_split=j, min_samples_leaf=k)
                clf.fit(X, y)
                y_pred = clf.predict(X)
                live.log_metric('Precision', precision_score(y, y_pred, average='micro'))
                live.log_metric('Recall', recall_score(y, y_pred, average='micro'))
                live.log_sklearn_plot("confusion_matrix", y, y_pred)
                conf_matrix = confusion_matrix(y, predictions)
                sns_plot = sns.heatmap(conf_matrix, annot=True)
                results_path = 'results.png'
                plt.savefig(results_path)
                live.log_image(f"img/{live.step}.png", 'results.png')
                live.next_step()