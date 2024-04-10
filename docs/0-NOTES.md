# Notes

Notes on process for reconstruction later.

## 2 Apr 2024

Got datasets from Kaggle.
Working idea to predict two things:

1. Overall freedom index
2. Change in freedom indices from 2022 to 2023

Use electricity generation trends and water+sanitation trends as inputs.

## 4 Apr 2024

Created project, and `data-raw/`, and extracted/converted all to CSV in `data-raw/csv/`.
Used Excel to convert the `.xlsx` file, then saved that as a zip
Set up virtual env with Python 3.12 (add `deadsnakes` PPA, then install `python3.12` and `python3.12-dev`)
Created EDA notebook

### Observations

Freedom values are pretty flat in the middle, like a plateau (based on min/max, quantiles and std dev)
Electricity production probably needs to be per-capita or some other normalization method
Water data is non-standard layout. Will need to data wrangle into a more columnar shape
    - transform to 1 row per country-year, with all indicator as a column
2012-2022 is our possible data set for years of inputs - that's the overlap in water and econ data
 We have many features in the water dataset, so let's find 1+ that is uncorrelated with electricity generation

ELEC:
    - Net Electricity Production, Electricity isn't correlated to any other elec_df variable

TWO UNCORR WATER: 
    - WS_PPL_W-SM (people with access to safely managed water)
    - WS_HCF_W-B (healthcare facilities with basic water services)

## 8 Apr 2024

When joining the data, zero countries have all the data.
Many have just a couple rows, and the measurement of HCF_W-B is very spotty (mostly NaN).
    - And when it's measured, it's always the same. So we gotta toss it, unless we go with "is_present" and "raw_value" or comparison...
I'll need to handle missing values if I want to maximize the dataset length.
    - Could use `0` for missing, with an accompanying "was_present" variable?
The chart is all over with plotting absolute values.
    - Let's calculate and plot deltas instead

## 9 Apr 2024
Put deltas into columns
Added target variables for economic freedom index scores
Reshaped and exported "final" dataset

ANALYSIS - let's ask:

1. How well do changes in water sanitization and electricity generation metrics predict the Change from 2022 magnitude and direction (ie: value)? (continuous prediction)
2. Is there a relationship between 2023 Score absolute magnitude (ex: above 50) and the trend in water sanitization or electricity or both? (classifier problem)

Then we can work on more feature selection and interpretation.

ITERATION (LINEAR):

The delta vars aren't significant when I add the `X.mean` for missing data.
Not much better when I drop rows with missing data.

The sum of the percentage changes also aren't predictive.
```
    data["water_sum_change"] = data[water_cols].fillna(0).sum(axis=1)
    data["elec_sum_change"] = data[elec_cols].fillna(0).sum(axis=1)
```
I used a technique from class to auto-combine a bunch of factors and got something reasonable for "Change from 2022".
    - Tested for similar params, but models collapsed - not strong.

I tried also a recent value for raw percentage of sanitation in the linear regression, and it was okay for GovtIntegrity

ITERATION (CLASSIFIER):

FLEX TO DT for overall classification.
    - I did a similar thing with the decision tree model finding
    - This was fine, but overall low `n` means hard to get a good model
    - tested for similar params, but models collapsed - not strong.
    - 85th percentile worked just fine...

## 10 Apr 2024

Backfilled data from newest year into oldest
Updated models
Did presentation
