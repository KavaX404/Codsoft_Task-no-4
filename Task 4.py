# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample dataset of movies with title and genre information
data = {'Title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump'],
        'Genre': ['Drama', 'Crime, Drama', 'Action, Crime, Drama', 'Crime, Drama', 'Drama, Romance']}

# Create DataFrame from the sample data
movies_df = pd.DataFrame(data)

# Function to recommend movies based on user preferences
def recommend_movies(user_preferences, movies_df):
    # TF-IDF vectorizer to convert genre text into vectors
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(movies_df['Genre'])
    
    # Calculate similarity scores between user preferences and movie genres
    user_preferences_vector = tfidf.transform([user_preferences])
    cosine_similarities = linear_kernel(user_preferences_vector, genre_matrix).flatten()
    
    # Get indices of movies sorted by similarity scores
    movie_indices = cosine_similarities.argsort()[::-1]
    
    # Recommend top 5 movies based on similarity scores
    recommended_movies = movies_df.iloc[movie_indices][:5]
    return recommended_movies

# Example: Recommend movies for a user interested in crime and drama
user_preferences = 'Romance, Drama'
recommended_movies = recommend_movies(user_preferences, movies_df)
print("Recommended movies based on user preferences:", recommended_movies)
