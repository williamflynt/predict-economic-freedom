from typing import Any

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from statsmodels.regression.linear_model import RegressionResults
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from src.data import get_dataset, ARTIFACT_DIR
from src.viz import plot_linear_regression

IS_ECONOMICALLY_FREE_DEFAULT_THRESHOLD = 66  # p75


def build_linear_econ_predictor(
    data: pd.DataFrame, formula: str = None
) -> tuple[sm.OLS, RegressionResults]:
    """
    Predict the change in Economic Freedom Index score from 2022 to 2023, given
    changes in water sanitation scores and electricity generation.

    :param data: pd.DataFrame containing features and targets.
    :return: tuple Fitted model and fit results.
    """
    if formula is None:
        # Below worked best before backfilling data.
        # formula = "Change_from_2022 ~ water_delta_3_yr + water_delta_3_yr:elec_delta_3_yr + water_delta_3_yr:delta_Net_Electricity_Production_Electricity_GWh_2022"
        formula = "Change_from_2022 ~ delta_WS_PPL_W_SM_2021:elec_delta_10_yr + delta_WS_PPL_W_SM_2021 + delta_WS_PPL_W_SM_2021:water_delta_10_yr:elec_delta_10_yr + delta_WS_PPL_W_SM_2021:water_delta_10_yr"
    model = smf.ols(formula=formula, data=data, missing="drop")
    results = model.fit()
    return model, results


def build_econ_classifier(
    data: pd.DataFrame,
    threshold: int = IS_ECONOMICALLY_FREE_DEFAULT_THRESHOLD,
    print_results: bool = True,
) -> tuple[DecisionTreeClassifier, Any, dict[str, float | Any]]:
    """
    Predict whether a country is "economically free" according to a threshold score,
    given trends in water sanitation scores and electricity production.

    :param data: pd.DataFrame containing features and targets.
    :param threshold: int Threshold for determining economic freedom binary.
    :param print_results: bool Whether to print model stats.
    :return: tuple Fitted model and results.
    """
    data["is_economically_free"] = (data["2023_Score"] > threshold).astype(int)

    X = data[["delta_WS_PPL_W_SM_2022", "water_delta_10_yr", "elec_delta_5_yr"]]
    y = data["is_economically_free"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "auc": roc_auc_score(y_test, predictions),
    }

    if print_results:
        print(
            "\n=========================\n| Decision Tree Results |\n========================="
        )
        print("Accuracy :", f"{accuracy_score(y_test, predictions):.3f}")
        print("Precision:", f"{precision_score(y_test, predictions):.3f}")
        print("Recall   :", f"{recall_score(y_test, predictions):.3f}")
        print("F1 Score :", f"{f1_score(y_test, predictions):.3f}")
        print("AUC      :", f"{roc_auc_score(y_test, predictions):.3f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

    return clf, predictions, results


if __name__ == "__main__":
    data = get_dataset()

    # Change from 2022
    change_model, change_results = build_linear_econ_predictor(data)
    print(change_results.summary())
    plot_linear_regression(data, change_model, change_results, "Change_from_2022")

    # Govt Integrity
    govt_model, govt_results = build_linear_econ_predictor(
        data, "Govt_Integrity ~ elec_delta_10_yr + WS_PPL_W_SM_2021:elec_delta_10_yr"
    )
    print(govt_results.summary())
    plot_linear_regression(data, govt_model, govt_results, "Govt_Integrity")

    # Is this country very "economically free"?
    tree, _, _ = build_econ_classifier(data)
    # Save the plot!
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree,
        filled=True,
        rounded=True,
        class_names=["Not Economically Free", "Economically Free"],
        feature_names=[
            "delta_WS_PPL_W_SM_2022",
            "water_delta_10_yr",
            "elec_delta_5_yr",
        ],
    )
    plt.savefig(ARTIFACT_DIR / "DecisionTree.png")
