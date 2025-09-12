# -*- coding: utf-8 -*-
"""
include: 
1. flights per destination within continental US (lon > -130) 
2. JFK: daily avg wind speed vs daily avg departure delay (joined tables weather+flights)
3. monthly total capacity by airlines
4. Boeing 737 vs Airbus A318/319/320/321 flights by origin
5. January hours where any two of JFK/EWR/LGA differ in temp by > 10°F
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

def load_nycflights13():
    # returns flights, weather, planes, airports, airlines as pandas DataFrames.
    import nycflights13 as nf
    flights = nf.flights()
    weather = nf.weather()
    planes = nf.planes()
    airports = nf.airports()
    airlines = nf.airlines()
    return flights, weather, planes, airports, airlines


# flights per destination (continental US)
def flights_per_dest_continental(flights, airports):
    # join flights with airports on dest == faa
    merged = flights.merge(airports, left_on="dest", right_on="faa", how="inner", suffixes=("", "_ap"))
    # filter to continental US (lon > -130)
    merged = merged.loc[merged["lon"] > -130]
    n_per_dest = (
        merged.groupby("dest", as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .merge(airports[["faa", "lat", "lon"]], left_on="dest", right_on="faa", how="left")
        [["dest", "lat", "lon", "n"]]
    )
    return n_per_dest


def plot_us_destinations(n_per_dest):
    bins = [0, 5000, 10000, 15000, 20000]
    labels = ["less than 5k", "5k-10k", "10k-15k", "15k-20k"]
    n_per_dest["count_interval"] = pd.cut(n_per_dest["n"], bins=bins, labels=labels, include_lowest=True)

    if HAS_CARTOPY:
        proj = ccrs.LambertConformal()
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=proj)
        ax.set_extent([-125, -66.5, 24, 49], ccrs.Geodetic())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.2)
        for lab in n_per_dest["count_interval"].cat.categories:
            sub = n_per_dest[n_per_dest["count_interval"] == lab]
            ax.scatter(sub["lon"], sub["lat"], transform=ccrs.Geodetic(), s=10, label=str(lab))
        ax.legend(title="Count interval", loc="lower left")
        ax.set_title("Number of flights for each destination in continental U.S.")
        plt.show()
    else:
        plt.figure(figsize=(9, 6))
        for lab in n_per_dest["count_interval"].cat.categories:
            sub = n_per_dest[n_per_dest["count_interval"] == lab]
            plt.scatter(sub["lon"], sub["lat"], s=10, label=str(lab))
        plt.xlabel("lon")
        plt.ylabel("lat")
        plt.title("Number of flights for each destination in continental U.S. (no basemap)")
        plt.legend(title="Count interval", loc="best")
        plt.show()


# JFK: daily avg wind speed vs daily avg dep delay
def jfk_wind_vs_delay(weather, flights):
    # join by (year, month, day, hour, origin), keep only shared rows, how=inner
    jfk_wx = weather[weather["origin"] == "JFK"].merge(
        flights, on=["year", "month", "day", "hour", "origin"], how="inner"
    )
    daily = (
        jfk_wx.groupby(["year", "month", "day"], as_index=False)
        .agg(daily_avg_wind_speed=("wind_speed", "mean"),
             daily_avg_departure_delay=("dep_delay", "mean"))
    )
    # scatter plot 
    plt.figure()
    plt.scatter(daily["daily_avg_wind_speed"], daily["daily_avg_departure_delay"], s=12)
    # line trend line 
    x = daily["daily_avg_wind_speed"].to_numpy()
    y = daily["daily_avg_departure_delay"].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 2:
        coef = np.polyfit(x[mask], y[mask], 1)
        xline = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        yline = coef[0] * xline + coef[1]
        plt.plot(xline, yline)
    plt.title("Daily Wind Speed vs Daily Avg Departure Delay at JFK")
    plt.xlabel("Daily Average Wind Speed (mph)")
    plt.ylabel("Daily Average Departure Delay (minutes)")
    plt.show()


# capacity by airline 
# only focus on non-cancelled ones, and set NA be 50 here 
INTERESTED_AIRLINES = {
    "American Airlines Inc.",
    "Delta Air Lines Inc.",
    "JetBlue Airways",
    "US Airways Inc.",
    "United Air Lines Inc.",
}

def monthly_capacity_by_airline(flights, airlines, planes):
    # join flights with airlines (carrier->name)
    f = flights.merge(airlines, on="carrier", how="inner")
    f = f[f["name"].isin(INTERESTED_AIRLINES)]
    f = f[~f["arr_delay"].isna()]
    f = f.merge(planes[["tailnum", "seats"]], on="tailnum", how="left")
    f["seats"] = f["seats"].fillna(50)
    cap = (
        f.groupby(["name", "month"], as_index=False)
         .agg(n=("seats", "sum"))
         .sort_values(["name", "month"])
    )

    for name, sub in cap.groupby("name"):
        plt.figure()
        plt.bar(sub["month"].astype(str), sub["n"])
        plt.title(f"Total Capacity Across 12 Months (2013) — {name}")
        plt.xlabel("Month")
        plt.ylabel("Total Capacity")
        plt.tight_layout()
        plt.show()

    return cap


# Boeing 737 vs Airbus A318/319/320/321 by origin
AIRBUS_PAT = r"^A318|^A319|^A320|^A321"
BOEING_PAT = r"^737"

def boeing_vs_airbus_by_origin(flights, planes):
    planes_sub = planes[planes["model"].astype(str).str.contains(BOEING_PAT + "|" + AIRBUS_PAT, regex=True, na=False)].copy()
    def label_manufacturer(model):
        m = str(model)
        if re.match(BOEING_PAT, m):
            return "Boeing 737"
        elif re.match(AIRBUS_PAT, m):
            return "Airbus A320 family"
        return np.nan

    planes_sub["manufacturer_group"] = planes_sub["model"].apply(label_manufacturer)
    planes_sub = planes_sub.dropna(subset=["manufacturer_group"])

    merged = planes_sub.merge(flights, on="tailnum", how="inner")
    counts = (
        merged.groupby(["origin", "manufacturer_group"], as_index=False)
              .size()
              .rename(columns={"size": "num"})
    )
    origins = counts["origin"].unique().tolist()
    groups = ["Boeing 737", "Airbus A320 family"]

    plt.figure()
    x = np.arange(len(origins))
    width = 0.35
    for i, g in enumerate(groups):
        sub = counts[counts["manufacturer_group"] == g].set_index("origin").reindex(origins).fillna(0)
        plt.bar(x + (i - 0.5) * width, sub["num"].to_numpy(), width=width, label=g)
    plt.xticks(x, origins)
    plt.xlabel("Origin")
    plt.ylabel("Number of Flights")
    plt.title("Number of Flights by Origin and Manufacturer")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return counts

# January hours with >10°F differences among JFK/EWR/LGA
def january_temp_diffs_gt10(flights, weather):
    # join, and only keep useful columns 
    merged = (
        flights[flights["origin"].isin(["JFK", "EWR", "LGA"])]
        .merge(weather, on=["year", "month", "day", "hour", "origin"], how="inner")
    )
    merged = merged[merged["month"] == 1]

    def agg_hour(g):
        temps = g.groupby("origin")["temp"].mean()
        tJ = temps.get("JFK", np.nan)
        tE = temps.get("EWR", np.nan)
        tL = temps.get("LGA", np.nan)
        return pd.Series({
            "temp_diff_JFK_EWR": abs(tJ - tE),
            "temp_diff_JFK_LGA": abs(tJ - tL),
            "temp_diff_EWR_LGA": abs(tE - tL),
        })

    diffs = (
        merged.groupby(["year", "month", "day", "hour"], as_index=False)
        .apply(agg_hour)
        .reset_index(drop=True)
    )

    gt10 = diffs[
        (diffs["temp_diff_JFK_EWR"] > 10)
        | (diffs["temp_diff_JFK_LGA"] > 10)
        | (diffs["temp_diff_EWR_LGA"] > 10)
    ].copy()

    print("January hours with >10°F difference among any two of JFK/EWR/LGA:")
    print(gt10)
    print("\nCount:", len(gt10))
    return gt10



def main():
    flights, weather, planes, airports, airlines = load_nycflights13()

    # 1. flights per destination (continental US) + map
    n_per_dest = flights_per_dest_continental(flights, airports)
    plot_us_destinations(n_per_dest)

    # 2. JFK daily wind vs delay
    jfk_wind_vs_delay(weather, flights)

    # 3. capacity by airline (non-cancelled; seats NA->50)
    _cap = monthly_capacity_by_airline(flights, airlines, planes)

    # 4. Boeing 737 vs Airbus A320 family by origin
    _counts = boeing_vs_airbus_by_origin(flights, planes)

    # 5. January temp diffs > 10°F
    _gt10 = january_temp_diffs_gt10(flights, weather)


if __name__ == "__main__":
    main()

