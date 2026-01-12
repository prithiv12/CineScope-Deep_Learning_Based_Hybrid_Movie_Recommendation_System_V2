import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

# =====================================================
# GLOBAL MODEL STORAGE
# =====================================================
USER_FACTORS = None
ITEM_FACTORS = None

USER_INDEX = {}
ITEM_INDEX = {}

# =====================================================
# TRAIN COLLABORATIVE FILTERING MODEL
# =====================================================
def train_collaborative_model(ratings_path="ratings.csv", n_factors=50):
    """
    Trains collaborative filtering model using matrix factorization (SVD).

    Returns:
        train_rmse (float): RMSE on training data
        test_rmse (float): RMSE on test data
    """

    global USER_FACTORS, ITEM_FACTORS
    global USER_INDEX, ITEM_INDEX

    # -------------------------------
    # LOAD & VALIDATE DATA
    # -------------------------------
    ratings = pd.read_csv(ratings_path)

    required_cols = {"userId", "movieId", "rating"}
    if not required_cols.issubset(ratings.columns):
        raise ValueError(
            "ratings.csv must contain userId, movieId, and rating columns"
        )

    # -------------------------------
    # TRAIN–TEST SPLIT
    # -------------------------------
    train_df, test_df = train_test_split(
        ratings,
        test_size=0.2,
        random_state=42
    )

    # -------------------------------
    # INDEX MAPPING (TRAIN ONLY)
    # -------------------------------
    user_ids = train_df["userId"].unique()
    movie_ids = train_df["movieId"].unique()

    USER_INDEX = {u: i for i, u in enumerate(user_ids)}
    ITEM_INDEX = {m: i for i, m in enumerate(movie_ids)}

    train_df["user_idx"] = train_df["userId"].map(USER_INDEX)
    train_df["movie_idx"] = train_df["movieId"].map(ITEM_INDEX)

    # -------------------------------
    # BUILD USER–ITEM MATRIX
    # -------------------------------
    R = csr_matrix(
        (train_df["rating"],
         (train_df["user_idx"], train_df["movie_idx"])),
        shape=(len(USER_INDEX), len(ITEM_INDEX))
    )

    # -------------------------------
    # MATRIX FACTORIZATION (SVD)
    # -------------------------------
    n_components = min(n_factors, R.shape[1] - 1)
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )

    USER_FACTORS = svd.fit_transform(R)
    ITEM_FACTORS = svd.components_.T

    # -------------------------------
    # TRAIN RMSE
    # -------------------------------
    train_preds, train_actuals = [], []

    for _, row in train_df.iterrows():
        u = row["user_idx"]
        i = row["movie_idx"]
        train_preds.append(
            np.dot(USER_FACTORS[u], ITEM_FACTORS[i])
        )
        train_actuals.append(row["rating"])

    train_rmse = np.sqrt(
        mean_squared_error(train_actuals, train_preds)
    )

    # -------------------------------
    # TEST RMSE (GENERALIZATION)
    # -------------------------------
    test_preds, test_actuals = [], []

    for _, row in test_df.iterrows():
        u = USER_INDEX.get(row["userId"])
        i = ITEM_INDEX.get(row["movieId"])

        # Ignore cold-start users/items
        if u is None or i is None:
            continue

        test_preds.append(
            np.dot(USER_FACTORS[u], ITEM_FACTORS[i])
        )
        test_actuals.append(row["rating"])

    test_rmse = np.sqrt(
        mean_squared_error(test_actuals, test_preds)
    )

    return train_rmse, test_rmse


# =====================================================
# PREDICT RATING (USED BY HYBRID MODEL)
# =====================================================
def predict_rating(user_id, movie_id):
    """
    Predicts rating for a given user and movie.
    Returns None for cold-start cases.
    """
    if user_id not in USER_INDEX or movie_id not in ITEM_INDEX:
        return None

    return float(
        np.dot(
            USER_FACTORS[USER_INDEX[user_id]],
            ITEM_FACTORS[ITEM_INDEX[movie_id]]
        )
    )
