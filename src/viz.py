import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.regression.linear_model import RegressionResults, OLS

from src.data import ARTIFACT_DIR


def plot_linear_regression(
    data: pd.DataFrame,
    model: OLS,
    results: RegressionResults,
    y_var: str,
    show: bool = False,
    save: bool = True,
):
    num_plots = len(model.exog_names) - 1  # subtract 1 for the intercept
    num_rows = int(num_plots**0.5)
    num_cols = num_plots // num_rows + (num_plots % num_rows > 0)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 6))

    # Create a new DataFrame for plotting that includes interaction terms
    plot_data = data.copy()
    for col in model.exog_names:
        if ":" in col:  # interaction term
            var1, var2 = col.split(":")
            plot_data[col] = data[var1] * data[var2]

    for i, col in enumerate(model.exog_names):
        if col == "Intercept":
            continue
        if num_rows == 1 or num_cols == 1:
            ax = axs[(i - 1) % num_cols]  # 1-dimensional indexing
        else:
            ax = axs[(i - 1) // num_cols, (i - 1) % num_cols]  # 2-dimensional indexing
        ax.plot(plot_data[col], results.params[i] * plot_data[col], label=col)
        ax.scatter(plot_data[col], plot_data[y_var], color="gray")
        ax.set_xlabel(col)
        ax.set_ylabel(y_var)
        ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig(ARTIFACT_DIR / f"{y_var}-Regression.png")
    if show:
        plt.show()
