import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix

USER_FACTORS = None
ITEM_FACTORS = None
RMSE_SCORE = None

USER_INDEX = {}
ITEM_INDEX = {}

def train_collaborative_model(ratings_path="ratings.csv", n_factors=50):
    global USER_FACTORS, ITEM_FACTORS, RMSE_SCORE
    global USER_INDEX, ITEM_INDEX

    ratings = pd.read_csv(ratings_path)
    required_cols = {"userId", "movieId", "rating"}
    if not required_cols.issubset(ratings.columns):
        raise ValueError("ratings.csv must contain userId, movieId, rating columns")

    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    USER_INDEX = {u: i for i, u in enumerate(user_ids)}
    ITEM_INDEX = {m: i for i, m in enumerate(movie_ids)}

    ratings["user_idx"] = ratings["userId"].map(USER_INDEX)
    ratings["movie_idx"] = ratings["movieId"].map(ITEM_INDEX)

    R = csr_matrix(
        (ratings["rating"],
         (ratings["user_idx"], ratings["movie_idx"])),
        shape=(len(USER_INDEX), len(ITEM_INDEX))
    )

    n_components = min(n_factors, R.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    USER_FACTORS = svd.fit_transform(R)
    ITEM_FACTORS = svd.components_.T

    preds, actuals = [], []
    for _, row in ratings.iterrows():
        u, i = row["user_idx"], row["movie_idx"]
        preds.append(np.dot(USER_FACTORS[u], ITEM_FACTORS[i]))
        actuals.append(row["rating"])

    RMSE_SCORE = np.sqrt(mean_squared_error(actuals, preds))
    return RMSE_SCORE


def predict_rating(user_id, movie_id):
    if user_id not in USER_INDEX or movie_id not in ITEM_INDEX:
        return None
    return float(np.dot(USER_FACTORS[USER_INDEX[user_id]],
                        ITEM_FACTORS[ITEM_INDEX[movie_id]]))
