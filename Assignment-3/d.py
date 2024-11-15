import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import pandas as pd
from implicit.als import AlternatingLeastSquares
import math
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt


class RecommenderEvaluator:
    def __init__(self, user_actor_matrix, test_size=0.2, random_state=42):
        """
        Initialize the evaluator with the user-actor matrix.
        """
        self.user_actor_matrix = user_actor_matrix
        self.test_size = test_size
        self.random_state = random_state
        self.train_matrix = None
        self.test_matrix = None
        self.model = None
        np.random.seed(random_state)
        
    def prepare_train_test_split(self):
        """Split the data into training and testing sets."""
        print("Preparing train-test split...")
        matrix_array = self.user_actor_matrix.values
        mask = np.random.rand(*matrix_array.shape) < (1 - self.test_size)
        
        train = matrix_array * mask
        test = matrix_array * ~mask
        
        self.train_matrix = pd.DataFrame(
            train, 
            index=self.user_actor_matrix.index,
            columns=self.user_actor_matrix.columns
        )
        self.test_matrix = pd.DataFrame(
            test,
            index=self.user_actor_matrix.index,
            columns=self.user_actor_matrix.columns
        )
        
        return csr_matrix(train), csr_matrix(test)
    
    def train_model(self, factors=50, regularization=0.1, iterations=20):
        """Train the ALS model on the training data."""
        print("Training the ALS model...")
        train_sparse, _ = self.prepare_train_test_split()
        
        with threadpool_limits(limits=1, user_api='blas'):
            self.model = AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                iterations=iterations
            )
            self.model.fit(train_sparse)
        
        print("Model training completed.")
        return self.model
    
    def get_recommendations(self, user_id, k=10):
        """Get recommendations for a user."""
        user_idx = self.train_matrix.index.get_loc(user_id)
        user_items = csr_matrix(self.train_matrix.loc[user_id].values.reshape(1, -1))
        
        try:
            recommended_ids, scores = self.model.recommend(
                userid=user_idx,
                user_items=user_items,
                N=k,
                filter_already_liked_items=True
            )
            return recommended_ids, scores
        except Exception as e:
            print(f"Error getting recommendations for user {user_id}: {str(e)}")
            return [], []
    
    def precision_at_k(self, user_id, k=10):
        """Calculate Precision@k for a user."""
        actual = set(self.test_matrix.columns[self.test_matrix.loc[user_id] > 0].tolist())
        
        if not actual:
            return 0.0
        
        recommended_ids, _ = self.get_recommendations(user_id, k)
        
        if len(recommended_ids) == 0:
            return 0.0
            
        predicted = set(self.train_matrix.columns[recommended_ids].tolist())
        
        return len(actual.intersection(predicted)) / k if k > 0 else 0.0
    
    def recall_at_k(self, user_id, k=10):
        """Calculate Recall@k for a user."""
        actual = set(self.test_matrix.columns[self.test_matrix.loc[user_id] > 0].tolist())
        
        if not actual:
            return 0.0
        
        recommended_ids, _ = self.get_recommendations(user_id, k)
        
        if len(recommended_ids) == 0:
            return 0.0
            
        predicted = set(self.train_matrix.columns[recommended_ids].tolist())
        
        return len(actual.intersection(predicted)) / len(actual) if len(actual) > 0 else 0.0
    
    def ndcg_at_k(self, user_id, k=10):
        """Calculate NDCG@k for a user."""
        actual_ratings = self.test_matrix.loc[user_id]
        actual_ratings = actual_ratings[actual_ratings > 0]
        
        if len(actual_ratings) == 0:
            return 0.0
        
        recommended_ids, scores = self.get_recommendations(user_id, k)
        
        if len(recommended_ids) == 0:
            return 0.0
        
        dcg = 0.0
        for i, item_id in enumerate(recommended_ids):
            item_name = self.train_matrix.columns[item_id]
            if item_name in actual_ratings.index:
                dcg += actual_ratings[item_name] / np.log2(i + 2)
        
        ideal_ratings = sorted(actual_ratings.values, reverse=True)[:k]
        idcg = sum(rating / np.log2(i + 2) for i, rating in enumerate(ideal_ratings))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def visualize_top_rated_actors(evaluator, user_id, k=10):
        """Visualize the top-rated actors for a given user."""
        recommended_ids, scores = evaluator.get_recommendations(user_id, k)
        actor_names = evaluator.train_matrix.columns[recommended_ids]
        
        plt.figure(figsize=(10, 6))
        plt.barh(actor_names, scores, color='skyblue')
        plt.xlabel('Scores')
        plt.title(f'Top {k} Recommended Actors for User {user_id}')
        plt.gca().invert_yaxis()
        plt.show()

    def evaluate_model(self, k=10, sample_size=None):
        """Evaluate the model using all metrics."""
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        
        users = self.user_actor_matrix.index.tolist()
        if sample_size and sample_size < len(users):
            users = np.random.choice(users, size=sample_size, replace=False)
        
        metrics = {
            f'precision@{k}': [],
            f'recall@{k}': [],
            f'ndcg@{k}': []
        }
        
        print("Evaluating model...")
        for user_id in tqdm(users, desc="Evaluating users"):
            try:
                metrics[f'precision@{k}'].append(self.precision_at_k(user_id, k))
                metrics[f'recall@{k}'].append(self.recall_at_k(user_id, k))
                metrics[f'ndcg@{k}'].append(self.ndcg_at_k(user_id, k))
            except Exception as e:
                print(f"Error evaluating user {user_id}: {str(e)}")
                continue
        
        return {
            metric: np.mean(scores) for metric, scores in metrics.items()
        }

def main():
    # Load data
    print("Loading data...")
    user_actor_matrix = pd.read_csv('user_actor_ratings.csv', index_col=0)
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = RecommenderEvaluator(user_actor_matrix)
    
    # Train model
    print("Training model...")
    evaluator.train_model()
    
    # Evaluate model
    # print("\nEvaluating model...")
    # Use a smaller sample size for faster evaluation during testing
    metrics = evaluator.evaluate_model(k=10, sample_size=20000)  # Adjust sample_size as needed
    
    # Print results
    print("\nEvaluation Results:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    # Visualize top-rated actors for a user
    user_id = 5  # Replace with actual user ID
    evaluator.visualize_top_rated_actors(user_id, k=10)





if __name__ == "__main__":
    main()
    



   
    