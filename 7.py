# -*- coding: utf-8 -*-
"""
Data: 
- HR_Analytics.csv  (IBM fictional HR dataset)
- Harry Potter note:
  Provide chapter lists for each book as Python lists of strings (one string per chapter).
  See the TODO sections below (placeholders named *_chapters).
"""

import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# HR Analytics
def load_hr(path="HR_Analytics.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def convert_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [
        "Attrition", "BusinessTravel", "Department", "EducationField",
        "Gender", "JobRole", "MaritalStatus", "Over18", "OverTime"
    ]
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def plot_attrition_age_gender(df: pd.DataFrame):
  # boxplot of employee attrition considering age and gender, colored by gender, side-by-side boxplot 
    needed = {"Attrition", "Age", "Gender"}
    if not needed.issubset(df.columns):
        print("Missing columns for attrition-age-gender plot; skipping.")
        return

    sub = df.dropna(subset=["Attrition", "Age", "Gender"]).copy()
    genders = sorted(sub["Gender"].astype(str).unique())
    attr_levels = sorted(sub["Attrition"].astype(str).unique())

    for g in genders:
        gdf = sub[sub["Gender"].astype(str) == g]
        data = [gdf.loc[gdf["Attrition"].astype(str) == a, "Age"].dropna().to_numpy()
                for a in attr_levels]
        plt.figure()
        plt.boxplot(data, labels=attr_levels)
        plt.title(f"Distribution of Employee Attrition by Age — Gender: {g}")
        plt.xlabel("Attrition")
        plt.ylabel("Age")
        plt.tight_layout()
        plt.show()


def plot_jobsatisfaction_by_jobrole(df: pd.DataFrame):
    if not {"JobRole", "JobSatisfaction"}.issubset(df.columns):
        print("Missing columns for JobSatisfaction plot; skipping.")
        return

    grp = (
        df.groupby("JobRole", as_index=False)["JobSatisfaction"]
          .mean(numeric_only=True)
          .rename(columns={"JobSatisfaction": "mean_JobSatisfaction_JobRole"})
          .sort_values("mean_JobSatisfaction_JobRole")
    )

    plt.figure(figsize=(8, max(4, len(grp) * 0.25)))
    y = np.arange(len(grp))
    plt.scatter(grp["mean_JobSatisfaction_JobRole"], y)
    plt.yticks(y, grp["JobRole"])
    plt.gca().invert_yaxis()  # highest at top, like coord_flip with reorder
    plt.title("Mean JobSatisfaction Across JobRoles")
    plt.xlabel("Mean JobSatisfaction")
    plt.ylabel("JobRole")
    plt.tight_layout()
    plt.show()


def plot_attrition_rate_by_distance_and_travel(df: pd.DataFrame):
  # convert to categorical type first 
    needed = {"DistanceFromHome", "BusinessTravel", "Attrition"}
    if not needed.issubset(df.columns):
        print("Missing columns for distance/travel heatmap; skipping.")
        return

    out = df.copy()
    out["DistanceFromHome_cat"] = pd.cut(
        out["DistanceFromHome"],
        bins=[-np.inf, 10, 20, np.inf],
        labels=["Not_Far", "Acceptable", "Too_Far"]
    )
    # attrition rate = proportion of "Yes"
    out["AttritionYes"] = (out["Attrition"].astype(str).str.strip().str.lower() == "yes").astype(float)

    heat = (
        out.groupby(["DistanceFromHome_cat", "BusinessTravel"], as_index=False)["AttritionYes"]
           .mean()
           .rename(columns={"AttritionYes": "Attrition_rate"})
    )

    mat = heat.pivot(index="BusinessTravel", columns="DistanceFromHome_cat", values="Attrition_rate")
  
    plt.figure()
    im = plt.imshow(mat.to_numpy(), aspect="auto")
    plt.colorbar(im, label="Attrition Rate")
    plt.xticks(range(len(mat.columns)), mat.columns)
    plt.yticks(range(len(mat.index)), mat.index)
    plt.title("Attrition Rate Across DistanceFromHome and BusinessTravel")
    plt.xlabel("Distance from Home")
    plt.ylabel("Business Travel")
    plt.tight_layout()
    plt.show()


# words with even number of 'e'
def words_with_even_e(words):
    out = []
    for w in words:
        cnt = w.lower().count("e")
        if cnt % 2 == 0:
            out.append(w)
    return out


# sentiment analysis (AFINN) — helpers
def load_afinn_df():
    # load and return dataframe 
    url = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-111.txt"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    # Each line: word<TAB>score
    data = [line.strip().split("\t") for line in r.text.splitlines() if line.strip()]
    afinn = pd.DataFrame(data, columns=["word", "value"])
    afinn["value"] = pd.to_numeric(afinn["value"], errors="coerce")
    afinn["word"] = afinn["word"].astype(str)
    return afinn


def tokenize_words(text: str) -> list:
    return re.findall(r"[A-Za-z']+", text.lower())


def chapter_sentiment_scores(chapters, afinn_df: pd.DataFrame) -> pd.DataFrame:
    # build dict for fast lookup
    lex = dict(zip(afinn_df["word"], afinn_df["value"]))
    rows = []
    for i, chap in enumerate(chapters, start=1):
        toks = tokenize_words(chap)
        vals = [lex.get(w) for w in toks if w in lex]
        if len(vals) == 0:
            score = np.nan
        else:
            score = float(np.mean(vals))
        rows.append({"chapter": i, "sentiment_score": score})
    return pd.DataFrame(rows)


def plot_chapter_sentiment(scores_df: pd.DataFrame, title: str):
    if scores_df.empty:
        print("No sentiment scores to plot.")
        return
    
    plt.figure()
    plt.plot(scores_df["chapter"], scores_df["sentiment_score"], marker="o")
    # Simple smoothing via moving average (window=3)
    s = scores_df["sentiment_score"].to_numpy(dtype=float)
  
    if np.isfinite(s).sum() >= 3:
        kernel = np.ones(3) / 3.0
        s_smooth = np.convolve(np.nan_to_num(s, nan=np.nanmean(s)), kernel, mode="same")
        plt.plot(scores_df["chapter"], s_smooth)
    
    plt.title(title)
    plt.xlabel("Chapter")
    plt.ylabel("Sentiment Score")
    plt.tight_layout()
    plt.show()


def average_sentiment_by_book(book_to_chapters: dict, afinn_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for book, chapters in book_to_chapters.items():
        df = chapter_sentiment_scores(chapters, afinn_df)
        rows.append({"book": book, "sentiment_score": float(df["sentiment_score"].mean(skipna=True))})
    out = pd.DataFrame(rows).sort_values("sentiment_score", ascending=False)
    return out


def plot_book_sentiments(book_df: pd.DataFrame):
    if book_df.empty:
        print("No book sentiment data to plot.")
        return
    
    df = book_df.copy().sort_values("sentiment_score")
  
    plt.figure(figsize=(8, max(4, len(df)*0.4)))
    y = np.arange(len(df))
    plt.barh(y, df["sentiment_score"])
    plt.yticks(y, df["book"])
    plt.title("Average Sentiment Scores of Each Harry Potter Book")
    plt.xlabel("Average Sentiment Score")
    plt.tight_layout()
    plt.show()


def unique_words_with_score(book_to_chapters: dict, afinn_df: pd.DataFrame, score=5) -> pd.DataFrame:
    all_text = " ".join(["\n".join(chapters) for chapters in book_to_chapters.values()])
    toks = set(tokenize_words(all_text))
    wanted = afinn_df[afinn_df["value"] == score]["word"]
    words = sorted(list(toks.intersection(set(wanted))))
    return pd.DataFrame({"word": words})



def main():
    # 1. HR Analytics 
    try:
        hr = load_hr("HR_Analytics.csv")
    except FileNotFoundError:
        print("HR_Analytics.csv not found. Place it next to this script.")
        hr = None

    if hr is not None:
        hr2 = convert_to_categorical(hr)
        plot_attrition_age_gender(hr2)
        plot_jobsatisfaction_by_jobrole(hr2)
        plot_attrition_rate_by_distance_and_travel(hr2)

    # 2. words with even number of 'e' 
    demo_words = ["tree", "cheese", "seem", "seen", "bee", "book", "letter", "example", "sky"]
    even_e = words_with_even_e(demo_words)
    print("\nWords with even number of 'e':", even_e)

    # 3. sentiment (Harry Potter) 
    afinn = load_afinn_df()

    # single-book analysis (Chamber of Secrets)
    if chamber_chapters:
        chamber_scores = chapter_sentiment_scores(chamber_chapters, afinn)
        print("\nChamber of Secrets sentiment (head):")
        print(chamber_scores.head())
        plot_chapter_sentiment(chamber_scores,
                               "Sentiment Changes over 19 Chapters of the Second Book in the Harry Potter Series")

    # all books
    books = {
        "Philosopher's Stone": [],
        "Chamber of Secrets": chamber_chapters,
        "Prisoner of Azkaban": [],
        "Goblet of Fire": [],
        "Order of the Phoenix": [],
        "Half-Blood Prince": [],
        "Deathly Hallows": [],
    }

    # compute iff at least one book has content
    if any(len(v) > 0 for v in books.values()):
        all_books_sent = average_sentiment_by_book(books, afinn)
        print("\nAverage sentiment by book:")
        print(all_books_sent)
        plot_book_sentiments(all_books_sent)

        # unique words with score = 5 across all books
        uniq_5 = unique_words_with_score(books, afinn, score=5)
        print("\nUnique words with AFINN score 5:")
        print(uniq_5.head())
    else:
        print("\n[Sentiment] Provide chapter texts to compute book-level results.")

if __name__ == "__main__":
    main()
