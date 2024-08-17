import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample data
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['The Matrix', 'Titanic', 'The Godfather', 'The Shawshank Redemption', 'Jurassic Park'],
    'genres': ['Action Sci-Fi', 'Romance Drama', 'Crime Drama', 'Drama', 'Action Adventure']
}

# Create DataFrame
df = pd.DataFrame(data)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the 'genres' column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['genres'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    # Check if the movie is in the dataset
    if title not in df['title'].values:
        return f"Movie '{title}' not found in the dataset."

    # Get the index of the movie that matches the title
    idx = df.index[df['title'] == title].tolist()[0]
    
    # Get pairwise similarity scores of all movies with the specified movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 3 most similar movies (excluding the input movie itself)
    sim_scores = sim_scores[1:4]
    
    # Get the indices of the top 3 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 3 most similar movies
    return df['title'].iloc[movie_indices].tolist()

# Test the recommendation system
if __name__ == "__main__":
    test_movie = 'The Matrix'
    print(f"Recommendations for '{test_movie}':")
    recommendations = get_recommendations(test_movie)
    for rec in recommendations:
        print(f"- {rec}")
