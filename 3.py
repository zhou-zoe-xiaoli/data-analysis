# -*- coding: utf-8 -*-
"""
Datasets:
- Detroit crime CSV: ./data/RMS_Crime_Incidents.csv.gz
- AirQuality: loaded from local airquality.csv if present, otherwise fetched from R-datasets URL.
"""

import os
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CRIME_PATH = "./data/RMS_Crime_Incidents.csv.gz"

def load_crime(path: str = CRIME_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Crime dataset not found at: {path}")
    return pd.read_csv(path, compression="infer")


def load_airquality(local_path: str = "airquality.csv") -> pd.DataFrame:
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    # Fallback URL (raw CSV of R datasets)
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv"
    try:
        df = pd.read_csv(url)
        # The CSV from Rdatasets has an extra 'Unnamed: 0' index column; drop it if present
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df
    except Exception as e:
        raise RuntimeError(
            "Could not load airquality dataset. Provide a local airquality.csv or ensure internet."
        ) from e

# When crimes occur
def summarize_day_period(crime: pd.DataFrame) -> pd.DataFrame:
  # divide and create day_period
    if "hour_of_day" not in crime.columns:
        raise KeyError("Expected column 'hour_of_day' in crime data.")
    day_period = pd.cut(
        crime["hour_of_day"],
        bins=[-np.inf, 6, 9, 17, 20, np.inf],
        labels=["Night", "Non-Work Hours", "Work Hours", "Non-Work Hours", "Night"],
        right=True,
        include_lowest=True,
    )
    crime2 = crime.copy()
    crime2["day_period"] = day_period
    out = crime2.groupby("day_period", dropna=False).size().reset_index(name="total")
    return crime2, out

# Crime trend (for top 3 2016–2022)
def top3_neighborhoods(crime: pd.DataFrame) -> pd.DataFrame:
    if "neighborhood" not in crime.columns:
        raise KeyError("Expected column 'neighborhood' in crime data.")
    return (
        crime.groupby("neighborhood", dropna=False)
        .size()
        .reset_index(name="total")
        .sort_values("total", ascending=False)
        .head(3)
    )


def yearly_counts_for_top3(crime: pd.DataFrame, top3: pd.DataFrame) -> pd.DataFrame:
    for col in ["neighborhood", "year"]:
        if col not in crime.columns:
            raise KeyError(f"Expected column '{col}' in crime data.")
    top_names = top3["neighborhood"].tolist()
    sub = crime.loc[
        crime["neighborhood"].isin(top_names) & (crime["year"] >= 2016) & (crime["year"] <= 2022),
        ["neighborhood", "year"],
    ]
    out = (
        sub.groupby(["neighborhood", "year"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["neighborhood", "year"])
    )
    return out


def plot_yearly_trend_top3(df_counts: pd.DataFrame):
  # line chart for crimes for top-3 neighborhoods 
    if df_counts.empty:
        print("No data to plot for yearly trend.")
        return
    plt.figure()
    for name, sub in df_counts.groupby("neighborhood"):
        plt.plot(sub["year"], sub["count"], marker="o", label=name)
    plt.title("Crimes over the Years for Top 3 Crime Neighborhoods")
    plt.xlabel("Year")
    plt.ylabel("Number of Crimes")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

# Specific crime types — clustering by rounded lat/long
CRIMES_OF_INTEREST = {
    "AGGRAVATED ASSAULT",
    "ARSON",
    "ASSAULT",
    "HOMICIDE",
    "JUSTIFIABLE HOMICIDE",
    "KIDNAPPING",
    "SEX OFFENSES",
    "SEXUAL ASSAULT",
    "WEAPONS OFFENSES",
}


def crimes_of_interest_counts(crime: pd.DataFrame) -> pd.DataFrame:
    # coordinates round to 3 decimal places 
    needed = {"offense_category", "X", "Y"}
    if not needed.issubset(crime.columns):
        missing = needed - set(crime.columns)
        raise KeyError(f"Missing required columns: {missing}")

    sub = crime.loc[crime["offense_category"].isin(CRIMES_OF_INTEREST)].copy()
    sub["new_long"] = pd.to_numeric(sub["X"], errors="coerce").round(3)
    sub["new_lat"] = pd.to_numeric(sub["Y"], errors="coerce").round(3)
    out = (
        sub.groupby(["new_lat", "new_long"], dropna=False)
        .size()
        .reset_index(name="count")
        .dropna(subset=["new_lat", "new_long"])
    )
    return out


def plot_crimes_of_interest_scatter(counts_df: pd.DataFrame):
    # scatter plot 
    if counts_df.empty:
        print("No crimes-of-interest counts to plot.")
        return
    plt.figure()
    sc = plt.scatter(counts_df["new_long"], counts_df["new_lat"], c=counts_df["count"], s=15)
    plt.colorbar(sc, label="Count")
    plt.title("Scatter Plot of Number of Crimes of Interest")
    plt.xlabel("Longitude (rounded to 0.001)")
    plt.ylabel("Latitude (rounded to 0.001)")
    plt.tight_layout()
    plt.show()

def normalized_offense_counts(crime: pd.DataFrame) -> pd.DataFrame:
    if "offense_category" not in crime.columns:
        raise KeyError("Expected column 'offense_category' in crime data.")
    cat_counts = (
        crime.groupby("offense_category", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    c = cat_counts["count"].astype(float)
    denom = (c.max() - c.min()) if (c.max() - c.min()) != 0 else 1.0
    cat_counts["category_normalized"] = (c - c.min()) / denom
    return cat_counts


# Air Quality analysis
def airquality_monthly_means(air: pd.DataFrame) -> pd.DataFrame:
    needed = {"Month", "Temp", "Ozone"}
    if not needed.issubset(air.columns):
        missing = needed - set(air.columns)
        raise KeyError(f"AirQuality missing columns: {missing}")
    out = (
        air.groupby("Month", dropna=False)
        .agg(mean_temp=("Temp", lambda x: np.nanmean(pd.to_numeric(x, errors="coerce"))),
             mean_ozone=("Ozone", lambda x: np.nanmean(pd.to_numeric(x, errors="coerce"))))
        .reset_index()
        .sort_values("Month")
    )
    return out


def plot_airquality_monthly(means_df: pd.DataFrame):
    # line plot of temp and Ozone 
    plt.figure()
    plt.plot(means_df["Month"], means_df["mean_temp"], marker="o", label="avg_temp")
    plt.plot(means_df["Month"], means_df["mean_ozone"], marker="o", label="avg_ozone")
    plt.title("Monthly Temperature and Ozone (Means)")
    plt.xlabel("Month")
    plt.ylabel("Average Values")
    plt.legend(title="Feature", loc="best")
    plt.tight_layout()
    plt.show()

def main():
    # Load Detroit crime data
    crime = load_crime()

    # 1) When do crimes occur?
    crime2, period_counts = summarize_day_period(crime)
    print("\nCrimes by day_period:")
    print(period_counts)

    # 2) Crime trend — top 3 neighborhoods, yearly counts (2016–2022)
    top3 = top3_neighborhoods(crime)
    print("\nTop three neighborhoods by total crime:")
    print(top3)
    yearly_top3 = yearly_counts_for_top3(crime, top3)
    print("\nYearly counts (2016–2022) for top three neighborhoods:")
    print(yearly_top3.head())
    plot_yearly_trend_top3(yearly_top3)

    # 3) Crimes of interest — scatter by rounded grid cell
    coi_counts = crimes_of_interest_counts(crime)
    print("\nCrimes of interest — top 5 grid cells by count:")
    print(coi_counts.sort_values("count", ascending=False).head())
    plot_crimes_of_interest_scatter(coi_counts)

    # 4) Normalized offense-category counts
    norm_counts = normalized_offense_counts(crime)
    print("\nNormalized offense-category counts (head):")
    print(norm_counts.head())

    # 5) Air Quality — monthly means & line plot
    air = load_airquality()
    print("\nAirQuality (head):")
    print(air.head())
    month_means = airquality_monthly_means(air)
    print("\nMonthly means of Temp & Ozone:")
    print(month_means)
    plot_airquality_monthly(month_means)


if __name__ == "__main__":
    main()
