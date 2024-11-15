# Report

## Data Collection and Preprocessing

### Data Scraping
For the first part of the assignment, I scraped data from The Movie Database (TMDB) using their API. The goal was to extract actor information for each movie listed in the `movies_set1.csv` file. The scraping process involved:

- **API Requests**: Sending requests to the TMDB API to retrieve actor information for each movie. The API key was used to authenticate the requests, and the movie titles and release years were used as query parameters.
- **Data Extraction**: Parsing the API responses to extract relevant actor details. The responses included a list of actors for each movie, from which the top 5 actors were selected based on their billing order.
- **Data Storage**: Storing the extracted actor information in a structured format. The data was saved in a CSV file named `movies_with_actors.csv`, which included columns for movie IDs, titles, and the top actors.

### Challenges Faced
- **API Rate Limiting**: TMDB API has rate limiting, which required implementing delays between requests to avoid being blocked. This was handled by adding a sleep interval between consecutive API calls.
- **Data Inconsistency**: Some movies did not have complete actor information, which required handling missing data appropriately. In cases where actor information was missing, the movie was either excluded from the dataset or the missing data was filled from alternative sources.
- **Error Handling**: Network issues and API errors were handled using retry logic. If an API request failed, the script retried the request up to three times before logging an error and moving on to the next movie.

### Resulting Dataset
The resulting dataset, `movies_with_actors.csv`, contains movie IDs along with the corresponding actors for each movie. This dataset was then merged with `ratings_set1.csv` to create a comprehensive user-actor rating matrix.

## User-Actor Rating Construction

### Calculating User-Actor Ratings
To construct the user-actor rating matrix, I merged the data from `ratings_set1.csv` and `movies_with_actors.csv` based on the `movieId`. The process involved:

1. **Mapping Movies to Actors**: Creating a mapping of movies to their respective actors using the `movies_with_actors.csv` file. Each movie was associated with a list of top actors.
2. **Aggregating Ratings**: For each user, aggregating the ratings they gave to movies and assigning those ratings to the actors in those movies. If a user rated multiple movies with the same actor, the ratings were averaged. This was done by exploding the actor list for each movie and then grouping by user and actor to calculate the average rating.

The resulting user-actor rating matrix was saved in the `user_actor_ratings.csv` file. This matrix is a sparse matrix where rows represent users, columns represent actors, and the values represent the aggregated ratings.

## Algorithm Selection

### Justification
I chose the Alternating Least Squares (ALS) algorithm for the recommendation system. ALS is well-suited for collaborative filtering tasks and can handle large, sparse datasets efficiently. It works by factorizing the user-item interaction matrix into two lower-dimensional matrices, which can then be used to predict missing values. The key advantages of ALS include:

- **Scalability**: ALS can handle large datasets with millions of users and items.
- **Implicit Feedback**: ALS can be used with implicit feedback data, which is common in recommendation systems.
- **Parallelization**: ALS can be parallelized, making it efficient for large-scale computations.

### Implementation
The ALS algorithm was implemented using the `implicit` library in Python. The steps involved:

1. **Data Preparation**: Converting the user-actor rating matrix into a sparse matrix format suitable for the ALS algorithm. This involved filling missing values with zeros and converting the DataFrame to a Compressed Sparse Row (CSR) matrix.
2. **Model Training**: Training the ALS model on the prepared data with specified hyperparameters (e.g., number of factors, regularization, iterations). The model was trained using the `fit` method of the ALS class.
3. **Generating Recommendations**: Using the trained model to generate top-10 actor recommendations for each user based on their user ID. The `recommend` method of the ALS class was used to generate recommendations, which included filtering out actors the user had already rated.

The trained model was saved in a `.pkl` file for future use.

## Evaluation and Analysis

### Metrics
The performance of the recommendation system was evaluated using the following metrics:

- **Precision@k**: Measures the proportion of relevant actors in the top-k recommendations. Precision@k is calculated as the number of relevant actors in the top-k recommendations divided by k.
- **Recall@k**: Indicates the proportion of all relevant actors retrieved within the top-k recommendations. Recall@k is calculated as the number of relevant actors in the top-k recommendations divided by the total number of relevant actors.
- **NDCG@k (Normalized Discounted Cumulative Gain)**: Assesses the quality of ranking by considering the position of relevant actors in the recommended list. NDCG@k is calculated by comparing the discounted cumulative gain (DCG) of the recommended list to the ideal DCG (IDCG).

### Results
The evaluation results for the model were as follows:

- **Precision@10**: 0.5472
- **Recall@10**: 0.3780
- **NDCG@10**: 0.6308

### Analysis
The results indicate that the recommendation system performs reasonably well, with a precision of 54.72%, recall of 37.80%, and NDCG of 63.08%. These metrics suggest that the model is effective in recommending relevant actors to users, but there is still room for improvement.

- **Precision**: The precision score indicates that more than half of the top-10 recommended actors are relevant to the user, which is a positive outcome.
- **Recall**: The recall score indicates that the model retrieves a significant portion of the relevant actors, but there is still room for improvement.
- **NDCG**: The NDCG score indicates that the ranking quality of the recommendations is good, but not perfect.

### Strengths and Limitations
**Strengths**:
- The ALS algorithm is efficient and scalable, making it suitable for large datasets.
- The model provides personalized recommendations based on user preferences.
- The evaluation metrics provide a comprehensive assessment of the model's performance.

**Limitations**:
- The model's performance may be affected by the sparsity of the user-actor rating matrix. Sparse matrices can lead to less accurate recommendations.
- The evaluation metrics indicate that there is room for improvement in terms of recall and ranking quality. Further tuning of the model parameters and experimenting with different algorithms could improve performance.

## Conclusion
In this assignment, I successfully implemented a recommendation system using the ALS algorithm. The system was evaluated using precision, recall, and NDCG metrics, which provided insights into its performance. The results indicate that the model performs reasonably well, but there is still room for improvement. Future work could involve experimenting with different algorithms, further tuning the model parameters, and exploring additional data sources to enhance the recommendations.

(Optional) **Visualization**:
To visualize the top-rated actors for a sample user, we could create a bar chart showing the top-10 recommended actors and their corresponding scores. This would provide a clear and intuitive representation of the recommendations. Additionally, visualizing the distribution of actor ratings could provide insights into the data and help identify patterns or trends.