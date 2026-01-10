CineScope – Deep Learning Based Hybrid Movie Recommendation System

Project Description:
CineScope is a hybrid movie recommendation system developed as a minor project for the MCA (Artificial Intelligence) program at Amrita Vishwa Vidyapeetham. The system combines deep learning–based semantic content understanding with collaborative filtering to provide personalized and relevant movie recommendations.

The project enhances a traditional content-based recommender by integrating transformer-based sentence embeddings (SBERT) to capture contextual meaning from movie genres and descriptions. User preferences are learned using collaborative filtering on MovieLens rating data. The system is implemented as an interactive Streamlit web application with explainability and evaluation features.

--------------------------------------------------

Project Objectives:

• To design and implement a hybrid movie recommendation system  
• To apply deep learning techniques for semantic understanding of movie content  
• To learn user preferences from historical user–movie rating data  
• To combine relevance and personalization using hybrid score-level fusion  
• To evaluate the recommendation system using standard metrics  
• To provide an interactive and explainable recommendation interface  

--------------------------------------------------

System Architecture Overview:

The system consists of three major components:

1. Content-Based Recommendation (Deep Learning):
   Movie genres and descriptions are converted into dense semantic embeddings using Sentence-BERT (SBERT), a transformer-based language model. Semantic similarity between movies is computed using cosine similarity on these embeddings.

2. Collaborative Filtering:
   User preferences are learned using matrix factorization on the MovieLens user–movie ratings dataset. The model predicts how likely a user is to prefer a given movie and is evaluated using RMSE.

3. Hybrid Recommendation:
   Content relevance scores and user preference scores are combined using a weighted hybrid fusion strategy. The balance between content relevance and personalization is controlled using a user-adjustable parameter.

--------------------------------------------------

Hybrid Recommendation Logic:

For a given base movie and user ID, the final recommendation score is computed as:

HybridScore = α × ContentScore + (1 − α) × UserPreferenceScore

Where:
• ContentScore is derived from SBERT-based semantic similarity  
• UserPreferenceScore is predicted using collaborative filtering  
• α controls the trade-off between relevance and personalization  

--------------------------------------------------

Explainability and Evaluation:

• Similarity Heatmap:
  Visualizes cosine similarity between recommended movies using SBERT embeddings to provide explainability.

• Evaluation Metrics:
  – RMSE is used to evaluate collaborative filtering accuracy  
  – Precision@K and Recall@K are used to evaluate ranking quality  

--------------------------------------------------

Files Description:

app.py  
Contains the Streamlit web application, including user interaction, movie search, genre browsing, similar movies, hybrid recommendations, explainability visualizations, and evaluation metrics.

recommender_core.py  
Implements the core recommendation logic, including SBERT embedding generation, similarity computation, hybrid score calculation, heatmap support, and evaluation metrics.

collaborative_filtering.py  
Implements collaborative filtering using matrix factorization and computes RMSE for model evaluation.

moviedataset_downloadandpreprocessing.py  
Downloads and preprocesses movie metadata from IMDb and retrieves movie descriptions using the TMDB API with caching.

ratingsdataset.py  
Downloads the MovieLens dataset and converts it into ratings.csv for collaborative filtering.

movies_with_description.csv  
Processed dataset containing movie titles, genres, ratings, votes, release year, and descriptions.

ratings.csv  
MovieLens user–movie rating dataset used for collaborative filtering.

tmdb_cache.csv  
Cache file for TMDB API responses to reduce repeated API calls.

requirements.txt  
Lists all required Python dependencies.

--------------------------------------------------

Technologies and Libraries Used:

• Python  
• Streamlit  
• Sentence-BERT (SBERT)  
• PyTorch  
• scikit-learn  
• pandas, numpy  
• matplotlib, seaborn  
• TMDB API  

--------------------------------------------------

How to Run the Project:

1. Install dependencies:
   pip install -r requirements.txt

2. Run the Streamlit application:
   streamlit run app.py

3. Open the displayed local URL in a web browser.

--------------------------------------------------

Academic Note:

This project demonstrates the practical implementation of a modern hybrid recommender system by integrating deep learning–based content modeling, collaborative filtering, explainability, and evaluation metrics. The system aligns with standard recommender system methodologies taught in artificial intelligence coursework.
