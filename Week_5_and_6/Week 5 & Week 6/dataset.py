import pandas as pd
import numpy as np

# Define feature genres
GENRES = [
    "Action",
    "Comedy",
    "Romance",
    "Sci-Fi",
    "Drama",
    "Thriller",
    "Animation",
    "Fantasy"
]

# 45 Movies with normalized genre vectors [0.0 to 1.0]
MOVIE_DATA = [
    # Sci-Fi / Action / Superhero
    {"title": "Avengers", "genres": [1.0, 0.2, 0.0, 0.9, 0.3, 0.4, 0.0, 0.8], "year": 2012},
    {"title": "Iron Man", "genres": [1.0, 0.3, 0.1, 1.0, 0.2, 0.4, 0.0, 0.3], "year": 2008},
    {"title": "Thor", "genres": [0.9, 0.3, 0.2, 0.6, 0.2, 0.3, 0.0, 1.0], "year": 2011},
    {"title": "Captain America", "genres": [1.0, 0.1, 0.2, 0.7, 0.4, 0.5, 0.0, 0.3], "year": 2011},
    {"title": "Guardians of the Galaxy", "genres": [0.9, 0.8, 0.2, 1.0, 0.2, 0.3, 0.0, 0.7], "year": 2014},
    {"title": "Spider-Man: Into the Spider-Verse", "genres": [0.9, 0.6, 0.1, 0.8, 0.4, 0.4, 1.0, 0.6], "year": 2018},
    {"title": "The Dark Knight", "genres": [1.0, 0.0, 0.1, 0.3, 0.9, 1.0, 0.0, 0.1], "year": 2008},
    {"title": "Man of Steel", "genres": [0.9, 0.0, 0.2, 0.9, 0.5, 0.4, 0.0, 0.6], "year": 2013},
    
    # Sci-Fi / Space / Mind-Bending
    {"title": "Interstellar", "genres": [0.4, 0.0, 0.2, 1.0, 0.9, 0.7, 0.0, 0.3], "year": 2014},
    {"title": "Inception", "genres": [0.8, 0.0, 0.2, 1.0, 0.6, 0.9, 0.0, 0.4], "year": 2010},
    {"title": "The Matrix", "genres": [1.0, 0.0, 0.1, 1.0, 0.4, 0.8, 0.0, 0.3], "year": 1999},
    {"title": "Blade Runner 2049", "genres": [0.5, 0.0, 0.2, 1.0, 0.8, 0.8, 0.0, 0.2], "year": 2017},
    {"title": "Arrival", "genres": [0.2, 0.0, 0.1, 1.0, 0.9, 0.7, 0.0, 0.1], "year": 2016},
    
    # Romance / Drama
    {"title": "Titanic", "genres": [0.2, 0.0, 1.0, 0.0, 1.0, 0.3, 0.0, 0.0], "year": 1997},
    {"title": "The Notebook", "genres": [0.0, 0.1, 1.0, 0.0, 0.9, 0.1, 0.0, 0.0], "year": 2004},
    {"title": "La La Land", "genres": [0.0, 0.5, 0.9, 0.0, 0.8, 0.0, 0.0, 0.2], "year": 2016},
    {"title": "Pride & Prejudice", "genres": [0.0, 0.2, 1.0, 0.0, 0.9, 0.0, 0.0, 0.0], "year": 2005},
    {"title": "About Time", "genres": [0.0, 0.6, 0.9, 0.5, 0.7, 0.0, 0.0, 0.4], "year": 2013},
    {"title": "500 Days of Summer", "genres": [0.0, 0.7, 0.9, 0.0, 0.6, 0.0, 0.0, 0.0], "year": 2009},

    # Comedy
    {"title": "The Hangover", "genres": [0.2, 1.0, 0.1, 0.0, 0.2, 0.1, 0.0, 0.0], "year": 2009},
    {"title": "Superbad", "genres": [0.1, 1.0, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0], "year": 2007},
    {"title": "Jump Street 21", "genres": [0.8, 1.0, 0.1, 0.0, 0.2, 0.3, 0.0, 0.0], "year": 2012},
    {"title": "Step Brothers", "genres": [0.1, 1.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0], "year": 2008},
    {"title": "Crazy Stupid Love", "genres": [0.0, 0.9, 0.9, 0.0, 0.5, 0.0, 0.0, 0.0], "year": 2011},

    # Animation / Fantasy
    {"title": "Spirited Away", "genres": [0.3, 0.2, 0.2, 0.2, 0.6, 0.3, 1.0, 1.0], "year": 2001},
    {"title": "Toy Story", "genres": [0.3, 0.9, 0.1, 0.2, 0.5, 0.1, 1.0, 0.5], "year": 1995},
    {"title": "The Lion King", "genres": [0.4, 0.4, 0.2, 0.0, 0.8, 0.2, 1.0, 0.5], "year": 1994},
    {"title": "Shrek", "genres": [0.4, 0.9, 0.4, 0.0, 0.3, 0.1, 1.0, 0.9], "year": 2001},
    {"title": "Your Name", "genres": [0.2, 0.3, 0.9, 0.5, 0.8, 0.3, 1.0, 0.7], "year": 2016},
    {"title": "Coco", "genres": [0.2, 0.6, 0.2, 0.0, 0.8, 0.1, 1.0, 0.8], "year": 2017},

    # Fantasy / Adventure
    {"title": "Harry Potter and the Sorcerer's Stone", "genres": [0.5, 0.3, 0.1, 0.2, 0.4, 0.3, 0.0, 1.0], "year": 2001},
    {"title": "The Lord of the Rings: The Fellowship", "genres": [0.9, 0.1, 0.2, 0.1, 0.8, 0.5, 0.0, 1.0], "year": 2001},
    {"title": "The Hobbit: An Unexpected Journey", "genres": [0.8, 0.3, 0.1, 0.1, 0.4, 0.4, 0.0, 1.0], "year": 2012},
    {"title": "Pirates of the Caribbean", "genres": [0.9, 0.7, 0.3, 0.1, 0.4, 0.4, 0.0, 0.8], "year": 2003},

    # Thriller / Crime / Drama
    {"title": "Pulp Fiction", "genres": [0.6, 0.5, 0.1, 0.0, 0.9, 0.9, 0.0, 0.0], "year": 1994},
    {"title": "Fight Club", "genres": [0.6, 0.2, 0.1, 0.3, 0.9, 1.0, 0.0, 0.0], "year": 1999},
    {"title": "Se7en", "genres": [0.3, 0.0, 0.0, 0.0, 0.9, 1.0, 0.0, 0.0], "year": 1995},
    {"title": "The Silence of the Lambs", "genres": [0.2, 0.0, 0.0, 0.0, 0.9, 1.0, 0.0, 0.0], "year": 1991},
    {"title": "Shutter Island", "genres": [0.2, 0.0, 0.1, 0.2, 0.9, 1.0, 0.0, 0.2], "year": 2010},

    # Pure Drama
    {"title": "The Shawshank Redemption", "genres": [0.1, 0.1, 0.1, 0.0, 1.0, 0.4, 0.0, 0.0], "year": 1994},
    {"title": "Forrest Gump", "genres": [0.2, 0.6, 0.6, 0.0, 0.9, 0.1, 0.0, 0.0], "year": 1994},
    {"title": "The Godfather", "genres": [0.5, 0.0, 0.2, 0.0, 1.0, 0.8, 0.0, 0.0], "year": 1972},
    {"title": "Whiplash", "genres": [0.1, 0.1, 0.1, 0.0, 1.0, 0.8, 0.0, 0.0], "year": 2014},
    {"title": "Oppenheimer", "genres": [0.3, 0.0, 0.1, 0.3, 1.0, 0.7, 0.0, 0.0], "year": 2023},
]


def load_movie_df():
    """Returns a pandas DataFrame containing titles, release years, and genre features."""
    records = []
    for item in MOVIE_DATA:
        row = {"Title": item["title"], "Year": item["year"]}
        for g, val in zip(GENRES, item["genres"]):
            row[g] = val
        records.append(row)
    return pd.DataFrame(records)


def get_movie_matrix():
    """
    Returns:
        titles (list): List of movie titles (N elements)
        X (np.ndarray): Movie-Feature Matrix of shape (N, d) where d=8
    """
    df = load_movie_df()
    titles = df["Title"].tolist()
    X = df[GENRES].to_numpy(dtype=np.float64)
    return titles, X


def get_movie_vector(movie_title):
    """Returns the feature vector for a given movie title."""
    df = load_movie_df()
    match = df[df["Title"].str.lower() == movie_title.lower()]
    if match.empty:
        raise ValueError(f"Movie '{movie_title}' not found in dataset.")
    vector = match[GENRES].to_numpy(dtype=np.float64)[0]
    return vector
