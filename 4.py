# -*- coding: utf-8 -*-
"""
Include: 
1. Custom functions: my_quantile, positive, no_outliers, summary-like output
2. Summaries using those functions
3. Visualizations for NYC flights carriers (top/bottom 10)
4. Starwars eye-color vs sex proportions + independence test
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_mpg() -> pd.DataFrame:
    try:
        from plotnine.data import mpg as gg_mpg
        return gg_mpg.copy()
    except Exception:
        pass

    try:
        import seaborn as sns
        df = sns.load_dataset("mpg").copy()
        if "displacement" in df.columns and "displ" not in df.columns:
            df["displ"] = df["displacement"]
        if "cty" not in df.columns and "mpg" in df.columns:
            df["cty"] = df["mpg"]
            df["hwy"] = df["mpg"]
        if "manufacturer" not in df.columns and "name" in df.columns:
            df["manufacturer"] = df["name"].astype(str).str.split().str[0]
        return df
    except Exception:
        raise RuntimeError(
            "Could not load 'mpg'"
        )


def load_starwars() -> pd.DataFrame:
    try:
        from vega_datasets import data as vdata
        url = vdata.url("starwars.json")
        return pd.read_json(url)
    except Exception:
        url = "https://raw.githubusercontent.com/vega/vega-datasets/master/data/starwars.json"
        return pd.read_json(url)


def load_flights() -> pd.DataFrame:
    try:
        import nycflights13 as nf
        return nf.flights()
    except Exception as e:
        raise RuntimeError(
            "Could not load nycflights13 flights"
        ) from e


# Custom functions for 90 quantile 
def my_quantile(x: pd.Series, quant: float) -> float:
    # 90 quantile 
    arr = pd.to_numeric(pd.Series(x).dropna(), errors="coerce").dropna().to_numpy()
    if arr.size == 0:
        return np.nan
    arr.sort()
    idx = int(math.ceil(quant * len(arr))) - 1  # convert to 0-based
    idx = max(0, min(idx, len(arr) - 1))
    return float(arr[idx])


def positive(x: pd.Series) -> bool:
    # predicate: returns True if all values are strictly > 0
    arr = pd.to_numeric(pd.Series(x), errors="coerce")
    return bool((arr.dropna() > 0).all())


def no_outliers(x: pd.Series) -> bool:
    # predicate: returns True if all observations are within 3 std dev of the mean
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if arr.size == 0:
        return True  
    mu = arr.mean()
    sd = arr.std(ddof=1)
    if sd == 0:
        return True
    return bool((np.abs(arr - mu) <= 3 * sd).all())


# summary and summarize 
def summary2(x: pd.Series) -> dict:
   # return a dict with Min, 1st Qu, Median, Mean, 3rd Qu, Max, Number Missing
   # using custom my_quantile for quartiles
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    vals = s.dropna()
    res = {
        "Min": float(vals.min()) if not vals.empty else np.nan,
        "1st Qu": my_quantile(vals, 0.25),
        "Median": float(vals.median()) if not vals.empty else np.nan,
        "Mean": float(vals.mean()) if not vals.empty else np.nan,
        "3rd Qu": my_quantile(vals, 0.75),
        "Max": float(vals.max()) if not vals.empty else np.nan,
        "Number Missing": int(s.isna().sum()),
    }
    return res


def summarize_all(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    names = ["Min", "1st Qu", "Median", "Mean", "3rd Qu", "Max", "Number Missing"]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == "object":
            stats = summary2(df[col])
            row = [stats[k] for k in names]
            out_rows.append((col, row))
    mat = pd.DataFrame([r for _, r in out_rows], columns=names, index=[c for c, _ in out_rows])
    return mat.round(5)


# visualizations
def plot_top_bottom_carriers(flights: pd.DataFrame):
    # top 10 and Bottom 10 carriers by frequency
    if "carrier" not in flights.columns:
        raise KeyError("Expected 'carrier' column in flights DataFrame.")

    carrier_count = flights.groupby("carrier").size().reset_index(name="total").sort_values("total")
    top_10 = carrier_count.tail(10)
    bottom_10 = carrier_count.head(10)

    print("top 10 carriers:", top_10["carrier"].tolist())
    print("bottom 10 carriers:", bottom_10["carrier"].tolist())

    # Top 10
    plt.figure()
    # order by total ascending on x-axis
    x = top_10.sort_values("total")["carrier"]
    y = top_10.sort_values("total")["total"]
    plt.bar(x, y)
    plt.title("Top 10 Carriers")
    plt.xlabel("Carrier")
    plt.ylabel("Frequency Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Bottom 10
    plt.figure()
    x = bottom_10.sort_values("total")["carrier"]
    y = bottom_10.sort_values("total")["total"]
    plt.bar(x, y)
    plt.title("Bottom 10 Carriers")
    plt.xlabel("Carrier")
    plt.ylabel("Frequency Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# starwars: eye_color proportions by sex + independence test
def starwars_eye_color_analysis():
    sw = load_starwars()

    # Clean: drop NA, keep only 'male'/'female', keep only black/blue/brown
    for col in ["sex", "eye_color"]:
        if col not in sw.columns:
            raise KeyError(f"starwars missing column: {col}")

    clean = sw.copy()
    clean = clean[clean["sex"].isin(["male", "female"])]
    clean = clean.dropna(subset=["sex", "eye_color"])
    clean = clean[clean["eye_color"].isin(["black", "blue", "brown"])]

    grp = (
        clean.groupby(["sex", "eye_color"])
        .size()
        .reset_index(name="count")
        .sort_values(["sex", "eye_color"])
    )

    totals_by_sex = grp.groupby("sex")["count"].transform("sum")
    grp["prop"] = grp["count"] / totals_by_sex

    pivot_prop = grp.pivot(index="sex", columns="eye_color", values="prop").fillna(0.0)
    pivot_counts = grp.pivot(index="sex", columns="eye_color", values="count").fillna(0.0)

    # plot 
    plt.figure()
    x = np.arange(len(pivot_prop.index))
    categories = ["black", "blue", "brown"]
    width = 0.25

    for i, color in enumerate(categories):
        vals = pivot_prop.get(color, pd.Series([0]*len(x), index=pivot_prop.index)).values
        plt.bar(x + (i - 1) * width, vals, width=width, label=color)

    plt.title("Distribution of Eye-color based on Sex (Proportions)")
    plt.xlabel("Sex")
    plt.ylabel("Proportion")
    plt.xticks(x, pivot_prop.index.tolist())
    plt.legend(title="eye_color")
    plt.tight_layout()
    plt.show()

    # independence test: use chi-square test for 2x3 table
    from scipy.stats import chi2_contingency

    contingency = pivot_counts.loc[["female", "male"], ["black", "blue", "brown"]].to_numpy()
    chi2, pval, dof, expected = chi2_contingency(contingency)
    print("Chi-square test of independence (sex vs eye_color among {black,blue,brown})")
    print(f"chi2={chi2:.4f}, dof={dof}, p-value={pval:.4g}")

    if pval > 0.05:
        print("Fail to reject H0 at α=0.05: no evidence of association between sex and eye color.")
    else:
        print("Reject H0 at α=0.05: evidence of association between sex and eye color.")


def main():
    mpg = load_mpg()
    # 90 quantitle 
    if {"manufacturer", "cty", "hwy"}.issubset(mpg.columns):
        grouped = []
        for manu, sub in mpg.groupby("manufacturer"):
            cty_90 = my_quantile(sub["cty"], 0.90)
            hwy_90 = my_quantile(sub["hwy"], 0.90)
            grouped.append((manu, cty_90, hwy_90))
        df_quant = pd.DataFrame(grouped, columns=["manufacturer", "cty_90", "hwy_90"])
        print("\nManufacturer-wise 90% quantiles (cty/hwy):")
        print(df_quant.sort_values("manufacturer").head())
    else:
        print("\nmpg missing expected columns for manufacturer/cty/hwy demo.")

    # generate random numbers and do tests 
    np.random.seed(2024306)
    n = 100
    d = pd.DataFrame({
        "x": np.random.uniform(size=n),
        "y": np.random.uniform(-1, 1, size=n),
        "z": np.random.normal(size=n) ** 2,
        "w": np.log(np.random.uniform(size=n)),
    })

    # select columns that are all positive 
    positive_cols = [col for col in d.columns if positive(d[col])]
    no_outlier_cols = [col for col in d.columns if no_outliers(d[col])]
    print("\nColumns where all values are strictly > 0:", positive_cols)
    print("Columns with no values beyond 3 SD from mean:", no_outlier_cols)

    # summarize and summarize_all
    d2 = d.copy()
    d2.iloc[9, 0] = np.nan
    d2.iloc[49:70, 1] = np.nan
    print("\nsummary2 for 'x':", summary2(d2["x"]))
    print("\nSummarize all columns:")
    print(summarize_all(d2))

    # NYC flights carriers (top & bottom 10)
    try:
        flights = load_flights()
        plot_top_bottom_carriers(flights)
    except Exception as e:
        print("\nSkipping flights charts:", e)

    # Starwars eye-color vs sex proportions + test
    starwars_eye_color_analysis()


if __name__ == "__main__":
    main()
