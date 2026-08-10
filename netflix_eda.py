"""
Netflix Titles Dataset - Exploratory Data Analysis (EDA)
Author: Ahsan Farooq

Explores the Netflix titles dataset to answer questions about dataset
structure, data quality, and content trends (Movies vs TV Shows, ratings,
genres, directors, and titles added per year).

Run with:  python netflix_eda.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "netflix_titles.csv")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

sns.set_theme(style="whitegrid")


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Step 1: Understanding the dataset
# ---------------------------------------------------------------------------
def explore_dataset(df):
    print("\n--- Step 1: Understanding the Dataset ---")

    print(f"\nShape (rows, columns): {df.shape}")

    print("\nFirst 5 entries:")
    print(df.head())

    print("\nUnique values per column:")
    print(df.nunique())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())


# ---------------------------------------------------------------------------
# Step 2: Data cleaning and preprocessing
# ---------------------------------------------------------------------------
def clean_data(df):
    print("\n--- Step 2: Data Cleaning and Preprocessing ---")

    # 'director', 'cast', 'country', and 'rating' contain nulls. Since the
    # missing share is under 50% and these are categorical/text fields
    # (mean/median/mode imputation would not be meaningful here), rows with
    # missing values are dropped rather than imputed.
    df = df.dropna().copy()

    print("\nUnique ratings:", df["rating"].unique())
    print("Unique durations (sample):", df["duration"].unique()[:10])
    print("Unique types:", df["type"].unique())
    print("Unique countries (sample):", df["country"].unique()[:10])

    # 'date_added' arrives as free-text (with stray leading whitespace on
    # some rows); convert to a proper datetime dtype so it can be used for
    # time-based aggregation later.
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip())

    print("\nData types after cleaning:")
    print(df.dtypes)

    return df


# ---------------------------------------------------------------------------
# Step 3: Exploring the dataset
# ---------------------------------------------------------------------------
def analyze_content_distribution(df):
    print("\n--- Content distribution (Movies vs TV Shows) ---")
    content_distribution = df.groupby("type").size()
    print(content_distribution)

    plt.figure(figsize=(6, 4))
    sns.countplot(x="type", data=df)
    plt.title("Content Distribution: Movies vs TV Shows")
    plt.xlabel("Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "content_distribution.png"))
    plt.close()

    return content_distribution


def analyze_ratings(df):
    print("\n--- Ratings distribution ---")
    unique_ratings = df["rating"].unique()
    print("Unique ratings:", unique_ratings)

    rating_distribution = df["rating"].value_counts()
    print(rating_distribution)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=rating_distribution.index, y=rating_distribution.values)
    plt.title("Distribution of Content Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "ratings_distribution.png"))
    plt.close()

    return rating_distribution


def analyze_genres(df):
    print("\n--- Unique genres ---")
    # 'listed_in' holds comma-separated genres per title, so titles can
    # belong to more than one genre. A set collects each distinct genre.
    unique_genres = set()
    for genres in df["listed_in"]:
        unique_genres.update(genres.split(", "))

    print(f"Total number of unique genres: {len(unique_genres)}")
    print("Unique genres:", sorted(unique_genres))

    return unique_genres


def analyze_directors(df):
    print("\n--- Titles per director ---")
    director_counts = df.groupby("director").size().sort_values(ascending=False)
    print(director_counts.head(10))
    return director_counts


def analyze_titles_per_year(df):
    print("\n--- Titles added to Netflix per year ---")
    df["year_added"] = df["date_added"].dt.year
    year_counts = df.groupby("year_added").size()
    print(year_counts)

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o")
    plt.title("Titles Added to Netflix per Year")
    plt.xlabel("Year Added")
    plt.ylabel("Number of Titles")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "titles_per_year.png"))
    plt.close()

    return year_counts


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    df = load_data()
    explore_dataset(df)

    df = clean_data(df)

    analyze_content_distribution(df)
    analyze_ratings(df)
    analyze_genres(df)
    analyze_directors(df)
    analyze_titles_per_year(df)

    print(f"\nCharts saved to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
