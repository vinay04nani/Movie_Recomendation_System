import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Calculate movie statistics
movies_df['average_rating'] = ratings_df.groupby('movieId')['rating'].mean()  # Average rating for each movie
movies_df['num_ratings'] = ratings_df.groupby('movieId')['rating'].count()    # Number of ratings for each movie
movies_df['popularity'] = movies_df['average_rating'] * movies_df['num_ratings']  # Popularity score based on average rating and number of ratings

# Plot distribution of movie ratings
plt.figure(figsize=(10, 6))
sns.histplot(ratings_df['rating'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Ratings')
plt.show()

# Plot top 10 most popular movies
top_10_popular = movies_df.sort_values('popularity', ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x='popularity', y='title', data=top_10_popular, palette='viridis')
plt.title('Top 10 Most Popular Movies')
plt.xlabel('Popularity Score')
plt.ylabel('Movie Title')
plt.show()

# Plot number of ratings per movie
plt.figure(figsize=(10, 6))
sns.histplot(movies_df['num_ratings'], bins=50, color='lightgreen')
plt.title('Number of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Count of Movies')
plt.show()

# Plot popularity vs. average rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='average_rating', y='popularity', data=movies_df, color='purple')
plt.title('Popularity vs. Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Popularity Score')
plt.show()

# Create a user-movie rating matrix
pivot_table = ratings_df.pivot(index='userId', columns='movieId', values='rating')
pivot_table.fillna(0, inplace=True)  # Replace NaN values with 0

# Convert pivot table to a sparse matrix
csr_matrix = csr_matrix(pivot_table.values)

# Initialize and train the KNN model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(csr_matrix)

# Find nearest neighbors for a given user
user_id = 18  # Replace with the desired user ID
known_neighbors = model.kneighbors(pivot_table.loc[user_id].values.reshape(1, -1), n_neighbors=5, return_distance=False)
known_neighbors = known_neighbors.flatten()

# Recommend movies based on nearest neighbors' ratings
recommended_movies = pivot_table.iloc[:, known_neighbors].mean(axis=1).sort_values(ascending=False)
recommended_movies = recommended_movies.index.tolist()

# Combine recommended movies with popularity data
recommended_movies_with_popularity = pd.merge(
    movies_df[['movieId', 'title', 'popularity']],
    pd.DataFrame({'movieId': recommended_movies}),
    on='movieId'
)
recommended_movies_with_popularity = recommended_movies_with_popularity.sort_values(by='popularity', ascending=False)
recommended_movies_with_popularity = recommended_movies_with_popularity.head(10)

# Display top 10 recommended movies for the user
print("Top 10 recommended movies for user ID 18:")
for index, row in recommended_movies_with_popularity.iterrows():
    movie_id = row['movieId']
    movie_name = row['title']
    movie_popularity = row['popularity']
    print(f"- {movie_id}: {movie_name} (Popularity: {movie_popularity})")
