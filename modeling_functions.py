import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc, ConfusionMatrixDisplay, \
    matthews_corrcoef, classification_report


# Provide model performance for binary classification
def model_dashboard_binary_classification(model, X_test, y_test):
    """Provide general model performance for binary classification with PR-Recall Curve and ConfusionMatrix plots.
    Prints accuracy, AUC PR-Recall Curve, AUC ROC Curve, MatthewsCorrCoef, classification report

    Args:
        model: fitted sklearn model that support .predict_proba()
        X_test (array-like): X for test (features)
        y_test (array-like): y for test (target)

    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    y_precision, y_recall, _ = precision_recall_curve(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    conf_mat = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=[0, 1])

    print("Accuracy:", model.score(X_test, y_test))
    print("AUC PR-Recall Curve:", auc(y_recall, y_precision))
    print("AUC ROC Curve:", auc(fpr, tpr))
    print("MCC:", matthews_corrcoef(y_test, y_pred))

    print("===" * 20)
    print(classification_report(y_test, y_pred, digits=6))
    print("===" * 20)

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))

    axs[0].set_title("PR-Recall Curve", fontsize=15)
    axs[0].plot(y_recall, y_precision)
    axs[1].set_title("ROC Curve", fontsize=15)
    axs[1].plot(fpr, tpr, color='darkorange', lw=2)
    axs[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axs[2].set_title("Confusion Matrix", fontsize=15)
    cm_display.plot(ax=axs[2])

    plt.show()
