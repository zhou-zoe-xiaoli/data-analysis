"""
Notes:
- The txhousing dataset is loaded from plotnine.data (same as ggplot2 in R).
- Faceting is replaced by generating one figure per city (one chart per figure).
- The Detroit crime CSV is expected at ./data/RMS_Crime_Incidents.csv.gz
- The Detroit map image (detroit.png) is provided in the local file.
"""

# import libararies 
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import txhousing dataset from plotnine.data
gg_txhousing = None
try:
    from plotnine.data import txhousing as gg_txhousing
except Exception:
    gg_txhousing = None


def load_txhousing() -> pd.DataFrame:
    # check install, and return error if not successful
    if gg_txhousing is None:
        raise RuntimeError("Please install plotnine.")
    return gg_txhousing.copy()


def rolling_smooth(y: pd.Series, window: int = 5) -> pd.Series:
    return y.rolling(window=window, center=True, min_periods=max(1, window // 2)).mean()


def plot_month_vs_median_scatter_with_smooth(df: pd.DataFrame, outpath: str = None):
    x = df["month"]
    y = df["median"]
    grouped = df.groupby("month")["median"].mean().sort_index()
    smooth_y = rolling_smooth(grouped, window=3)

    plt.figure()
    plt.scatter(x, y)
    plt.plot(grouped.index.values, smooth_y.values)
    plt.xlabel("month")
    plt.ylabel("median")
    plt.title("Median vs Month (points + smoothed trend)")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.show()


def plot_polar_month_vs_median(df: pd.DataFrame, outpath: str = None):
    # approximation of coord_polar
    months = df["month"].to_numpy()
    angles = (months - 1) / 12.0 * 2.0 * math.pi
    r = df["median"].to_numpy()

    plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.scatter(angles, r, s=10)
    ax.set_title("Polar plot: median by month")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.show()


def plot_median_over_time_colored_by_city(df: pd.DataFrame, outpath: str = None):
    plt.figure()
    for city, sub in df.groupby("city"):
        plt.plot(sub["date"], sub["median"], linestyle="", marker=".", label=city, alpha=0.6)
    plt.xlabel("date")
    plt.ylabel("median")
    plt.title("Median Prices Over Time by City (points)")
    plt.legend(loc="best", fontsize="small")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.show()


def plot_median_over_time_per_city(df: pd.DataFrame, max_cities: int = 8, outdir: str = None):
    cities = df["city"].dropna().unique()
    cities = sorted(cities)[:max_cities]

    for city in cities:
        sub = df[df["city"] == city]
        plt.figure()
        plt.plot(sub["date"], sub["median"])
        plt.xlabel("date")
        plt.ylabel("median")
        plt.title(f"Median over time — {city}")
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, f"median_over_time_{city}.png"), bbox_inches="tight", dpi=150)
        plt.show()


def plot_selected_cities_lines(df: pd.DataFrame, cities=None, outdir: str = None):
    if cities is None:
        cities = ["Galveston", "San Marcos", "South Padre Island"]

    for city in cities:
        sub = df[df["city"] == city]
        if sub.empty:
            continue
        plt.figure()
        plt.plot(sub["date"], sub["median"])
        plt.xlabel("date")
        plt.ylabel("median")
        plt.title(f"Median over time — {city}")
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, f"median_over_time_{city}.png"), bbox_inches="tight", dpi=150)
        plt.show()


def boxplot_sales_by_city(df: pd.DataFrame, outpath: str = None):
    # boxplot of sales by city
    cities = sorted(df["city"].dropna().unique())
    data = [df.loc[df["city"] == c, "sales"].dropna().to_numpy() for c in cities]

    plt.figure(figsize=(8, max(4, len(cities) * 0.2)))
    bp = plt.boxplot(data, vert=False, labels=cities, showfliers=False)
    
    means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in data]
    y_pos = np.arange(1, len(cities) + 1)
    
    plt.scatter(means, y_pos, s=10)
    plt.xlabel("sales")
    plt.ylabel("city")
    plt.title("Average Monthly Sales Variation for Each City in Texas")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.show()


def load_detroit_crime(path: str = "./data/RMS_Crime_Incidents.csv.gz") -> pd.DataFrame:
    # import detroit csv
    if not os.path.exists(path):
        raise FileNotFoundError(f"Crime dataset not found at: {path}")
    return pd.read_csv(path, compression='infer')


def basic_crime_stats(crime: pd.DataFrame):
    num_rows, num_cols = crime.shape
    print(f"There are {num_cols} columns and {num_rows} rows.")

    if "precinct" in crime.columns:
        num_precinct = crime["precinct"].nunique(dropna=True)
        print(f"There are {num_precinct} precincts.")
    else:
        print("Column 'precinct' not found.")


def plot_offense_category_distribution(crime: pd.DataFrame, outpath: str = None):
    # bar plot
    if "offense_category" not in crime.columns:
        print("Column 'offense_category' not found; skipping plot.")
        return
    counts = crime["offense_category"].value_counts(dropna=False)

    plt.figure(figsize=(8, max(4, len(counts) * 0.25)))
    plt.barh(counts.index.astype(str), counts.values)
    plt.xlabel("Number")
    plt.ylabel("Offense Category")
    plt.title("Distribution of Offense Category")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.show()


def offenses_at_address(crime: pd.DataFrame, address: str = "W Chicago St & Sussex"):
    # output a frequency table
    if "address" not in crime.columns or "offense_category" not in crime.columns:
        print("Required columns not found for address filter.")
        return
    sub = crime.loc[crime["address"] == address, "offense_category"]
    tbl = sub.value_counts(dropna=False)
    print(f"Offenses at '{address}':")
    print(tbl)


def plot_longitude_histogram(crime: pd.DataFrame, outpath: str = None):
    # histogram of longitudes
    col = "X"
    if col not in crime.columns:
        print(f"Column '{col}' not found; skipping histogram.")
        return
    x = pd.to_numeric(crime[col], errors="coerce").dropna().to_numpy()

    plt.figure()
    bins = int((np.nanmax(x) - np.nanmin(x)) / 0.002) if len(x) else 50
    bins = max(10, bins)
    plt.hist(x, bins=bins)
    plt.xlabel("Longitude (X)")
    plt.ylabel("Count")
    plt.title("Histogram of X (Longitude)")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.show()

    # Determine which of [83.1, 83.2, 83.0] is closest to the mode magnitude
    candidates = np.array([83.1, 83.2, 83.0])
    hist, edges = np.histogram(x, bins=bins)
    idx = np.argmax(hist)
    center = (edges[idx] + edges[idx + 1]) / 2.0
    closest = candidates[np.argmin(np.abs(np.abs(center) - candidates))]
    print(f"Most offenses happen near longitude magnitude: {abs(center):.3f}. Closest choice: {closest}.")


def density_on_map(crime: pd.DataFrame, map_path: str = "detroit.png", outpath: str = None):
    # 2D density plot overlaid on the Detroit map
    from PIL import Image

    if not os.path.exists(map_path):
        print(f"Map image not found at {map_path}; skipping density overlay.")
        return
    if "X" not in crime.columns or "Y" not in crime.columns:
        print("Columns 'X' and/or 'Y' not found; skipping density overlay.")
        return

    x = pd.to_numeric(crime["X"], errors="coerce").to_numpy()
    y = pd.to_numeric(crime["Y"], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    img = Image.open(map_path)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img, extent=[xmin, xmax, ymin, ymax], aspect='auto')
    plt.hist2d(x, y, bins=150, alpha=0.5)
    plt.xlabel("X (Longitude)")
    plt.ylabel("Y (Latitude)")
    plt.title("2D Density over Detroit Map")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.show()


def main():
    txhousing = load_txhousing()

    # plot: month vs median with smooth
    plot_month_vs_median_scatter_with_smooth(txhousing)

    # plot: polar version
    plot_polar_month_vs_median(txhousing)

    # plot: median over time, points colored by city
    plot_median_over_time_colored_by_city(txhousing)

    # plot: replacement for facet_wrap
    plot_median_over_time_per_city(txhousing, max_cities=8)

    # plot: selected cities 
    plot_selected_cities_lines(txhousing, cities=["Galveston", "San Marcos", "South Padre Island"])

    # plot: boxplot of sales by city
    boxplot_sales_by_city(txhousing)

    # crime one 
    try:
        crime = load_detroit_crime("./data/RMS_Crime_Incidents.csv.gz")
    except FileNotFoundError as e:
        print(e)
        return

    basic_crime_stats(crime)

    # distribution of offense_category
    plot_offense_category_distribution(crime)

    # offenses at specific address
    offenses_at_address(crime, address="W Chicago St & Sussex")

    # histogram of longitudes
    plot_longitude_histogram(crime)

    # 2D density overlay on map image
    density_on_map(crime, map_path="detroit.png")


if __name__ == "__main__":
    main()
