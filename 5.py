# -*- coding: utf-8 -*-
"""
Data: Storms, a built-in dataset in R, here transforms to Python 
Columns used (with fallbacks):
- category, year, month, day, name
- hurricane_force_diameter (or hu_diameter)
- wind, pressure
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOCAL_STORMS = "./data/storms.csv" # downloaded though original R file or online downloand it 
REMOTE_STORMS = "https://raw.githubusercontent.com/tidyverse/dplyr/main/data-raw/storms.csv"  # fallback


def load_storms() -> pd.DataFrame:
    if os.path.exists(LOCAL_STORMS):
        df = pd.read_csv(LOCAL_STORMS)
    else:
        # fallback to remote (requires internet)
        df = pd.read_csv(REMOTE_STORMS)
      
    if "hurricane_force_diameter" in df.columns:
        hu_col = "hurricane_force_diameter"
    elif "hu_diameter" in df.columns:
        hu_col = "hu_diameter"
    else:
        hu_col = "hurricane_force_diameter"
        df[hu_col] = np.nan

    needed = ["category", "year", "month", "day", "name", "wind", "pressure", hu_col]
    for col in needed:
        if col not in df.columns:
            # add as NA if missing
            df[col] = np.nan

  # type check and reasure 
    for c in ["year", "month", "day", "wind", "pressure", hu_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, hu_col


# dealn with missing values, NAs
def columns_with_missing(df: pd.DataFrame):
    cols = df.columns[df.isna().sum() > 0].tolist()
    print("Columns containing NA values:", cols)
    return cols

# ECDF plot 
def plot_ecdf(series: pd.Series, title: str, xlabel: str):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if x.size == 0:
        print("No data to plot ECDF.")
        return
    x_sorted = np.sort(x)
    y = np.arange(1, x_sorted.size + 1) / x_sorted.size

    plt.figure()
    plt.plot(x_sorted, y, marker=".", linestyle="none")
    plt.xlabel(xlabel)
    plt.ylabel("Proportion ≤ x")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# modifications for hurricane_force_diameter
def impute_hu_diameter(df: pd.DataFrame, hu_col: str) -> pd.DataFrame:
    out = df.copy()

    # dealing with NAs, one option 
    mask_zero = out["category"].isna() & out[hu_col].isna()
    out.loc[mask_zero, hu_col] = 0.0

    # for NAs where category is NOT NA, fill with mean by category
    means = (
        out.groupby("category", dropna=True)[hu_col]
        .mean()
        .rename("mean_hu_diam")
        .reset_index()
    )
    out = out.merge(means, on="category", how="left")
    mask_fill = out[hu_col].isna() & out["category"].notna()
    out.loc[mask_fill, hu_col] = out.loc[mask_fill, "mean_hu_diam"]
    out = out.drop(columns=["mean_hu_diam"], errors="ignore")
    return out


def hist_by_category(df: pd.DataFrame, hu_col: str, title: str):
    # one plot for each category 
    cats = df["category"].dropna().unique()
    for cat in sorted(cats, key=lambda z: (np.nan if pd.isna(z) else z)):
        sub = df.loc[df["category"] == cat, hu_col].dropna().to_numpy()
        if sub.size == 0:
            continue
        plt.figure()
        plt.hist(sub, bins=30)
        plt.title(f"{title} — category {cat}")
        plt.xlabel(hu_col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


# Frequency over time by category + linear trends
def frequency_over_time(df: pd.DataFrame):
    # n_distinct(name) per (category, year)
    grp = (
        df.dropna(subset=["category", "year"])
          .groupby(["category", "year"])["name"]
          .nunique()
          .reset_index(name="total")
          .sort_values(["category", "year"])
    )
    # plot with linear trend per category
    for cat, sub in grp.groupby("category"):
        plt.figure()
        plt.plot(sub["year"], sub["total"], marker="o", label=f"cat {cat}")
        # linear fit
        if len(sub) >= 2:
            coeff = np.polyfit(sub["year"], sub["total"], deg=1)
            x_line = np.linspace(sub["year"].min(), sub["year"].max(), 100)
            y_line = coeff[0] * x_line + coeff[1]
            plt.plot(x_line, y_line, label="Linear trend")
        plt.title(f"Frequency of Hurricanes by Category Over Time — cat {cat}")
        plt.xlabel("Year")
        plt.ylabel("Count (distinct storm names)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return grp


# days with highest number of unique storms
def add_date_and_top_days(df: pd.DataFrame):
    out = df.copy()
    out["date"] = pd.to_datetime(
        dict(year=out["year"], month=out["month"], day=out["day"]),
        errors="coerce"
    )
    counts = (
        out.groupby("date")["name"]
           .nunique()
           .reset_index(name="unique_storms")
           .sort_values("unique_storms", ascending=False)
    )
    print("Top days by unique storm names:")
    print(counts.head())
    return out, counts

# monthly frequency by category (in polar coordinates)
def monthly_frequency_polar(df: pd.DataFrame):
    sub = df.dropna(subset=["category", "year", "month"]).copy()
    sub["month"] = sub["month"].astype(int)
    freq = (
        sub.groupby(["category", "year", "month"])
           .size()
           .reset_index(name="frequency")
    )

    for cat, cat_df in freq.groupby("category"):
        # looking for seasonal patterns 
        monthly = (
            cat_df.groupby("month")["frequency"]
                  .sum()
                  .reindex(range(1, 13), fill_value=0)
        )
        # polar bar chart 
        angles = (monthly.index - 1) / 12 * 2 * math.pi
        plt.figure()
        ax = plt.subplot(111, projection="polar")
        ax.bar(angles, monthly.values, width=(2 * math.pi / 12))
        ax.set_title(f"Monthly Frequency (sum over years) — category {cat}")
        ax.set_xticks(np.linspace(0, 2 * math.pi, 12, endpoint=False))
        ax.set_xticklabels([str(m) for m in range(1, 13)])
        plt.tight_layout()
        plt.show()


# boxplot of wind variation by categories 
def wind_boxplot_by_category(df: pd.DataFrame):
    sub = df.dropna(subset=["wind", "category"]).copy()
    cats = sorted(sub["category"].dropna().unique())
    data = [sub.loc[sub["category"] == c, "wind"].values for c in cats]
    plt.figure()
    plt.boxplot(data, labels=[str(c) for c in cats])
    plt.title("Wind Speed Variation Across Hurricane Categories")
    plt.xlabel("Category")
    plt.ylabel("Wind Speed (knots)")
    plt.tight_layout()
    plt.show()


# wind verse pressure, colored by category
def wind_vs_pressure(df: pd.DataFrame):
    sub = df.dropna(subset=["wind", "pressure", "category"])
    if sub.empty:
        print("No data for wind vs pressure.")
        return
    cats = sorted(sub["category"].unique())
    plt.figure()
    for c in cats:
        sc = sub.loc[sub["category"] == c]
        plt.scatter(sc["wind"], sc["pressure"], s=10, label=str(c))
    plt.title("Max Sustained Wind vs Center Pressure")
    plt.xlabel("Wind (knots)")
    plt.ylabel("Pressure (millibars)")
    plt.legend(title="category", markerscale=2)
    plt.tight_layout()
    plt.show()



# correlation coefficients heatmap between each pair of variables 
def correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).dropna()
    if numeric.empty:
        print("No numeric data for correlation.")
        return
    corr = numeric.corr()
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlation")
    plt.title("Correlation Heatmap of Storm Parameters")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    plt.show()



def main():
    storms, hu_col = load_storms()

    # identify NAs
    columns_with_missing(storms)

    # ECDF for hurricane force diameter (non-missing)
    plot_ecdf(
        storms[hu_col].dropna(),
        title="ECDF of Hurricane Force Diameter",
        xlabel=hu_col
    )
    print(
        "Observation: If many values are zero, ECDF will jump near 0, supporting your R comment."
    )

    # compute hurricane diameters 
    storms2 = impute_hu_diameter(storms, hu_col)

    # histograms by category for the imputed column
    hist_by_category(storms2, hu_col, title="Distribution of Hurricane Force Diameter by Category")

    # frequency over time by category + trend lines
    freq_time = frequency_over_time(storms)

    # date and days with highest number of unique storms
    storms3, top_days = add_date_and_top_days(storms2)

    # monthly seasonal patterns per category in polar coordinates 
    monthly_frequency_polar(storms)

    # wind variation by category
    wind_boxplot_by_category(storms)

    # wind vs pressure scatter by category
    wind_vs_pressure(storms)

    # Correlation heatmap
    correlation_heatmap(storms)


if __name__ == "__main__":
    main()
