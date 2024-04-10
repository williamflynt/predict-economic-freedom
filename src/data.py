import os
import pathlib
import pandas as pd
import numpy as np
from collections import defaultdict

ARTIFACT_DIR = pathlib.Path(f"{os.getcwd()}/../artifacts/")
CSV_DIR = pathlib.Path(f"{os.getcwd()}/../data-raw/csv/")
assert pathlib.Path.exists(CSV_DIR), f"You must extract ZIP files to '{CSV_DIR}'"

# For calculating deltas.
MISSING_SENTINEL = np.nan
COUNTRY = 0
YEAR = 1
WATER = 2
ELEC = 3


def calculate_deltas(data: pd.DataFrame) -> dict:
    # Values like { <country_name>: { <year>: [<water_delta>, <elec_delta>] } }
    acc = defaultdict(dict)
    records = data.to_records(index=False)

    prev_country_name = ""
    prev_year_water = MISSING_SENTINEL
    prev_year_elec = MISSING_SENTINEL

    for record in records:
        acc[record[COUNTRY]][record[YEAR]] = [np.nan, np.nan]

        if record[COUNTRY] != prev_country_name:
            prev_country_name = record[COUNTRY]
            prev_year_water = record[WATER]
            prev_year_elec = record[ELEC]
            continue

        if prev_year_water != MISSING_SENTINEL and not np.isnan(record[WATER]):
            acc[record[COUNTRY]][record[YEAR]][0] = (
                (record[WATER] - prev_year_water) / prev_year_water
            ) * 100

        if prev_year_elec != MISSING_SENTINEL and not np.isnan(record[ELEC]):
            acc[record[COUNTRY]][record[YEAR]][1] = (
                (record[ELEC] - prev_year_elec) / prev_year_elec
            ) * 100

        prev_year_water = record[WATER]
        prev_year_elec = record[ELEC]

    return acc


def get_econ_data() -> pd.DataFrame:
    econ_df = pd.read_csv(CSV_DIR / "world_economy_freedom.csv")
    econ_df = econ_df.rename(columns={"Country Name": "country_name"})
    return econ_df


def get_elec_data() -> pd.DataFrame:
    elec_df = pd.read_csv(CSV_DIR / "global_electricity_production_data.csv")
    elec_df["year"] = elec_df["date"].apply(lambda x: int(x.split("/")[-1]))
    elec_pivot = elec_df.pivot_table(
        index=["country_name", "year"],
        columns=["parameter", "product", "unit"],
        values="value",
    )
    flat_elec_pivot = elec_pivot.copy(deep=True)
    flat_elec_pivot.columns = [
        "_".join(col).strip() for col in elec_pivot.columns.values
    ]
    flat_elec_pivot.reset_index(inplace=True)
    return flat_elec_pivot


def get_water_data() -> pd.DataFrame:
    water_df = pd.read_csv(CSV_DIR / "water.csv")
    water_pivot = water_df.pivot_table(
        index=["REF_AREA:Geographic area", "TIME_PERIOD:Time period"],
        columns="INDICATOR:Indicator",
        values="OBS_VALUE:Observation Value",
    ).reset_index()

    water_pivot.columns = [col.split(":")[0] for col in water_pivot.columns]
    water_pivot.TIME_PERIOD = water_pivot.TIME_PERIOD.astype(int)
    water_pivot.rename(
        columns={"REF_AREA": "country_name", "TIME_PERIOD": "year"}, inplace=True
    )
    water_pivot.country_name = (
        water_pivot.country_name.str.split(":").str[-1].str.strip()
    )
    return water_pivot


def get_raw_features(water_data: pd.DataFrame, elec_data: pd.DataFrame) -> pd.DataFrame:
    return water_data[["country_name", "year", "WS_PPL_W-SM"]].merge(
        elec_data[
            ["country_name", "year", "Net Electricity Production_Electricity_GWh"]
        ],
        how="inner",
        on=["country_name", "year"],
    )


def get_feature_deltas(raw_features: pd.DataFrame) -> pd.DataFrame:
    deltas = calculate_deltas(raw_features)
    # DELTAS to DataFrame.
    deltas_df = pd.DataFrame(deltas).T
    deltas_df.columns = pd.MultiIndex.from_tuples(
        [(year, "delta") for year in deltas_df.columns]
    )
    # UNPACK DELTAS.
    for year in deltas_df.columns.get_level_values(0).unique():
        deltas_df[("delta_WS_PPL_W-SM", year)] = deltas_df[(year, "delta")].apply(
            lambda x: x[0] if hasattr(x, "__iter__") else np.nan
        )
        deltas_df[
            ("delta_Net Electricity Production_Electricity_GWh", year)
        ] = deltas_df[(year, "delta")].apply(
            lambda x: x[1] if hasattr(x, "__iter__") else np.nan
        )
    # DROP OLD DELTAS OBJECTS.
    deltas_df = deltas_df.drop(columns="delta", level=1)
    return deltas_df


def get_final_features(
    raw_features: pd.DataFrame, deltas_data: pd.DataFrame
) -> pd.DataFrame:
    # FLATTEN RAW FEATURES - single row per country.
    df_pivot = raw_features.pivot_table(index="country_name", columns="year")
    # MERGE flattened data with deltas information.
    feature_df = pd.merge(
        df_pivot, deltas_data, left_index=True, right_index=True, how="outer"
    )
    # SORT and REPLACE missing values.
    feature_df = feature_df.sort_index(axis=1).replace({None: np.nan})
    # DROP known empty deltas columns.
    feature_df = feature_df.drop(
        columns=[
            ("delta_Net Electricity Production_Electricity_GWh", 2012),
            ("delta_WS_PPL_W-SM", 2012),
        ]
    )
    # RESHAPE FOR EXPORT
    feature_df.columns = [
        "_".join([str(part) for part in col]).strip()
        for col in feature_df.columns.values
    ]
    feature_df.reset_index(inplace=True)
    feature_df.rename(columns={"index": "country_name"}, inplace=True)
    return feature_df


def fill_na_2022(data: pd.DataFrame) -> pd.DataFrame:
    """
    This will put our delta values over time to zero for missing data,
    instead of 1.
    :param data:
    :return:
    """
    water_cols_to_fill = [col for col in data.columns if col.startswith("WS_PPL")]
    water_cols_to_fill = list(reversed(water_cols_to_fill))  # Most recent first.
    for i, c in enumerate(water_cols_to_fill[1:]):
        # Backfill from present to past in a cascade.
        data[c] = data[c].fillna(data[water_cols_to_fill[i - 1]])

    elec_cols_to_fill = [col for col in data.columns if col.startswith("Net_Elec")]
    elec_cols_to_fill = list(reversed(elec_cols_to_fill))  # Most recent first.
    for i, c in enumerate(elec_cols_to_fill[1:]):
        # Backfill from present to past in a cascade.
        data[c] = data[c].fillna(data[elec_cols_to_fill[i - 1]])

    return data


def with_multiyear_deltas(data: pd.DataFrame) -> pd.DataFrame:
    data = fill_na_2022(data)

    data["water_delta_10_yr"] = (
        data["WS_PPL_W_SM_2022"] - data["WS_PPL_W_SM_2012"]
    ) / data["WS_PPL_W_SM_2022"]
    data["water_delta_5_yr"] = (
        data["WS_PPL_W_SM_2022"] - data["WS_PPL_W_SM_2017"]
    ) / data["WS_PPL_W_SM_2022"]
    data["water_delta_4_yr"] = (
        data["WS_PPL_W_SM_2022"] - data["WS_PPL_W_SM_2018"]
    ) / data["WS_PPL_W_SM_2022"]
    data["water_delta_3_yr"] = (
        data["WS_PPL_W_SM_2022"] - data["WS_PPL_W_SM_2019"]
    ) / data["WS_PPL_W_SM_2022"]
    data["water_delta_2_yr"] = (
        data["WS_PPL_W_SM_2022"] - data["WS_PPL_W_SM_2020"]
    ) / data["WS_PPL_W_SM_2022"]
    data["elec_delta_10_yr"] = (
        data["Net_Electricity_Production_Electricity_GWh_2022"]
        - data["WS_PPL_W_SM_2012"]
    ) / data["Net_Electricity_Production_Electricity_GWh_2022"]
    data["elec_delta_5_yr"] = (
        data["Net_Electricity_Production_Electricity_GWh_2022"]
        - data["Net_Electricity_Production_Electricity_GWh_2017"]
    ) / data["Net_Electricity_Production_Electricity_GWh_2022"]
    data["elec_delta_4_yr"] = (
        data["Net_Electricity_Production_Electricity_GWh_2022"]
        - data["Net_Electricity_Production_Electricity_GWh_2018"]
    ) / data["Net_Electricity_Production_Electricity_GWh_2022"]
    data["elec_delta_3_yr"] = (
        data["Net_Electricity_Production_Electricity_GWh_2022"]
        - data["Net_Electricity_Production_Electricity_GWh_2019"]
    ) / data["Net_Electricity_Production_Electricity_GWh_2022"]
    data["elec_delta_2_yr"] = (
        data["Net_Electricity_Production_Electricity_GWh_2022"]
        - data["Net_Electricity_Production_Electricity_GWh_2020"]
    ) / data["Net_Electricity_Production_Electricity_GWh_2022"]
    return data


def join_targets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    on: str = "country_name",
    target_cols: list = None,
) -> pd.DataFrame:
    if target_cols is None:
        target_cols = [
            "Region",
            "Govt Integrity",
            "Gov't Spending",
            "Tax Burden",
            "2022 Score",
            "2023 Score",
            "Change from 2022",
        ]
    df = features.merge(targets[[on] + target_cols], on=on, how="left")
    columns = {
        col: "_".join(
            [part for part in col.replace("'", "").replace("-", "_").split(" ")]
        )
        for col in df.columns
    }
    return df.rename(columns=columns)


def get_dataset() -> pd.DataFrame:
    raw_features = get_raw_features(get_water_data(), get_elec_data())
    deltas_data = get_feature_deltas(raw_features)
    final_features = get_final_features(raw_features, deltas_data)
    data = join_targets(final_features, get_econ_data())
    return with_multiyear_deltas(data)


if __name__ == "__main__":
    df = get_dataset()
    print(df.sample(3))
