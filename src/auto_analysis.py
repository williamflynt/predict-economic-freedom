from typing import Any, List

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

from src.data import get_dataset

IS_ECONOMICALLY_FREE_DEFAULT_THRESHOLD = 66


def all_params_ok(results: RegressionResults) -> bool:
    for p in results.pvalues:
        if p > 0.05:
            return False
    return True


def prune_params(results: RegressionResults) -> list:
    p_values = results.pvalues
    # Remove "Intercept"
    if "Intercept" in p_values:
        p_values = p_values.drop("Intercept")
    # Sort in descending order
    sorted_p_values = p_values.sort_values(ascending=False)
    # Remove variable with largest p-value
    pruned_vars = sorted_p_values.index[1:].tolist()
    return pruned_vars


def build_linear_econ_predictor(
    data: pd.DataFrame,
    use_cols=None,
    predict: str = "Govt_Integrity",
    prune_attempts: int = 0,
) -> tuple[sm.OLS, RegressionResults]:
    """
    Predict the change in Economic Freedom Index score from 2022 to 2023, given
    changes in water sanitation scores and electricity generation.

    :param data: pd.DataFrame containing features and targets.
    :return: tuple Fitted model and fit results.
    """
    the_cols = use_cols or ["water_delta_5_yr", "water_delta_3_yr", "water_delta_2_yr"]
    joined_for_model = " * ".join(the_cols)
    formula = f"{predict} ~ {joined_for_model}"
    model = smf.ols(formula=formula, data=data, missing="drop")
    results = model.fit()
    while not all_params_ok(results) and prune_attempts > 0:
        prune_attempts -= 1
        new_params = prune_params(results)
        if not new_params:
            break
        new_joined = " + ".join(new_params)
        formula = f"{predict} ~ {new_joined}"
        model = smf.ols(formula=formula, data=data, missing="drop")
        results = model.fit()
    return model, results


def build_econ_classifier(
    data: pd.DataFrame,
    use_cols: List[str] = None,
    threshold: int = IS_ECONOMICALLY_FREE_DEFAULT_THRESHOLD,
    print_results: bool = True,
    test_size: float = 0.5,
) -> tuple[DecisionTreeClassifier, Any, dict[str, float | Any]]:
    """
    Predict whether a country is "economically free" according to a threshold score,
    given trends in water sanitation scores and electricity production.

    :param use_cols:
    :param test_size:
    :param data: pd.DataFrame containing features and targets.
    :param threshold: int Threshold for determining economic freedom binary.
    :param print_results: bool Whether to print model stats.
    :return: tuple Fitted model and results.
    """
    data["is_economically_free"] = (data["2023_Score"] > threshold).astype(int)

    if use_cols is None:
        use_cols = ["water_delta_3_yr", "elec_delta_3_yr"]

    X = data[use_cols]
    y = data["is_economically_free"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
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
    import itertools

    data = get_dataset()

    elec_single_deltas = [
        # "delta_Net_Electricity_Production_Electricity_GWh_2013",
        # "delta_Net_Electricity_Production_Electricity_GWh_2014",
        # "delta_Net_Electricity_Production_Electricity_GWh_2015",
        # "delta_Net_Electricity_Production_Electricity_GWh_2016",
        # "delta_Net_Electricity_Production_Electricity_GWh_2017",
        # "delta_Net_Electricity_Production_Electricity_GWh_2018",
        # "delta_Net_Electricity_Production_Electricity_GWh_2019",
        # "delta_Net_Electricity_Production_Electricity_GWh_2020",
        "delta_Net_Electricity_Production_Electricity_GWh_2021",
        "delta_Net_Electricity_Production_Electricity_GWh_2022",
    ]
    water_single_deltas = [
        # "delta_WS_PPL_W_SM_2013",
        # "delta_WS_PPL_W_SM_2014",
        # "delta_WS_PPL_W_SM_2015",
        # "delta_WS_PPL_W_SM_2016",
        # "delta_WS_PPL_W_SM_2017",
        # "delta_WS_PPL_W_SM_2018",
        # "delta_WS_PPL_W_SM_2019",
        # "delta_WS_PPL_W_SM_2020",
        "delta_WS_PPL_W_SM_2021",
        "delta_WS_PPL_W_SM_2022",
    ]
    water_single_values = [
        # "WS_PPL_W_SM_2012",
        # "WS_PPL_W_SM_2013",
        # "WS_PPL_W_SM_2014",
        # "WS_PPL_W_SM_2015",
        # "WS_PPL_W_SM_2016",
        # "WS_PPL_W_SM_2017",
        # "WS_PPL_W_SM_2018",
        # "WS_PPL_W_SM_2019",
        # "WS_PPL_W_SM_2020",
        "WS_PPL_W_SM_2021",
        "WS_PPL_W_SM_2022",
    ]
    water_cols = [
        "water_delta_10_yr",
        "water_delta_5_yr",
        "water_delta_4_yr",
        "water_delta_3_yr",
        "water_delta_2_yr",
    ]
    elec_cols = [
        "elec_delta_10_yr",
        "elec_delta_5_yr",
        "elec_delta_4_yr",
        "elec_delta_3_yr",
        "elec_delta_2_yr",
    ]
    single_deltas = water_single_deltas + elec_single_deltas
    combinations = list(
        itertools.product(single_deltas + water_single_values, water_cols, elec_cols)
    )

    best_model = None
    best_results = None
    for target in ["Change_from_2022", "Govt_Integrity"]:
        for combo in combinations:
            linear_model, linear_results = build_linear_econ_predictor(
                data,
                use_cols=list(combo),
                predict=target,
                prune_attempts=15,
            )
            if best_model is None or (
                linear_results.rsquared_adj > best_results.rsquared_adj
                and (all_params_ok(linear_results) or not all_params_ok(best_results))
            ):
                best_model = linear_model
                best_results = linear_results
                print(
                    f"{linear_results.model.formula}\n\tall significant: {all_params_ok(linear_results)}\n\tadj_r2: {linear_results.rsquared_adj}"
                )

    print(best_results.summary())
    print(best_results.model.formula)

    best_clf = None
    best_predictions = None
    best_auc_roc = -1
    best_cols = []
    best_test_size = 0.2
    best_results = {}
    for combo in combinations:
        for test_size in range(2, 7):
            size_float = test_size / 10
            clf, predictions, results = build_econ_classifier(
                data,
                use_cols=list(combo),
                test_size=size_float,
                print_results=False,
            )
            if results["auc"] > best_auc_roc:
                best_clf = clf
                best_predictions = predictions
                best_auc_roc = results["auc"]
                best_cols = combo
                best_test_size = size_float
                best_results = results
                print(
                    f"New best columns: {combo} @ {size_float}\n\tAUC-ROC: {best_auc_roc:.3f}"
                )
    print(
        f"Best columns: {best_cols} @ {best_test_size}\n\tAUC-ROC: {best_auc_roc:.3f}\n\tAccuracy: {best_results['accuracy']}"
    )
