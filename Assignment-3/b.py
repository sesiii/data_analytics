import pandas as pd

# Load the datasets
ratings_df = pd.read_csv('ratings_set1.csv')
movies_actors_df = pd.read_csv('movies_with_actors.csv')

# Merge the datasets on movieId
merged_df = pd.merge(ratings_df, movies_actors_df, on='movieId')

# Explode the top_actors column to have one actor per row
merged_df['top_actors'] = merged_df['top_actors'].str.split(', ')
exploded_df = merged_df.explode('top_actors')

# Group by userId and top_actors to calculate the average rating
user_actor_ratings = exploded_df.groupby(['userId', 'top_actors'])['rating'].mean().reset_index()

# Pivot the table to create the user-actor rating matrix
user_actor_matrix = user_actor_ratings.pivot(index='userId', columns='top_actors', values='rating')

# Fill NaN values with 0 (optional, depending on your use case)
user_actor_matrix = user_actor_matrix.fillna(0)

# Save the user-actor rating matrix to a CSV file
user_actor_matrix.to_csv('user_actor_ratings.csv')

print("User-actor rating matrix saved to 'user_actor_ratings.csv'")