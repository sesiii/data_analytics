import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import pickle
from threadpoolctl import threadpool_limits

# Set OpenBLAS to use single thread to avoid performance issues
with threadpool_limits(limits=1, user_api='blas'):
    
    def load_and_prepare_data(filepath):
        """Load and prepare the user-actor rating matrix."""
        # Load the user-actor rating matrix
        user_actor_matrix = pd.read_csv(filepath, index_col=0)
        
        # Convert the DataFrame to a sparse matrix format suitable for implicit
        user_actor_matrix = user_actor_matrix.fillna(0)
        
        # Convert to CSR matrix
        sparse_matrix = csr_matrix(user_actor_matrix.values)
        
        return user_actor_matrix, sparse_matrix

    def train_model(sparse_matrix, factors=100, regularization=0.05, iterations=30):
        """Train the ALS model."""
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations
        )
        
        # Train the model
        model.fit(sparse_matrix)
        return model

    def get_top_n_recommendations(model, user_id, user_actor_matrix, sparse_matrix, n=10):
        """Get top N actor recommendations for a user."""
        try:
            # Get the user's index
            if user_id not in user_actor_matrix.index:
                raise ValueError(f"User ID {user_id} not found in the dataset")
                
            user_index = user_actor_matrix.index.get_loc(user_id)
            
            # Get recommendations
            ids, scores = model.recommend(
                userid=user_index,
                user_items=sparse_matrix[user_index],
                N=n,
                filter_already_liked_items=True
            )
            
            # Get actor names if they're in your columns
            actor_names = user_actor_matrix.columns[ids].tolist()
            
            # Create recommendations dictionary
            recommendations = {
                'actor_indices': ids,
                'scores': scores,
                'actor_names': actor_names
            }
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return None

    def save_model(model, filepath='user_actor_model.pkl'):
        """Save the trained model to a file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model successfully saved to '{filepath}'")
        except Exception as e:
            print(f"Error saving model: {str(e)}")


    def main():
        try:
            # Load and prepare data
            user_actor_matrix, sparse_matrix = load_and_prepare_data('user_actor_ratings.csv')
            
            # Train model
            model = train_model(sparse_matrix)
            
            # Example usage
            user_id = 5  # Replace with actual user ID
            recommendations = get_top_n_recommendations(
                model,
                user_id,
                user_actor_matrix,
                sparse_matrix,
                n=10
            )
            
            if recommendations:
                print("\nTop 10 recommended actors:")
                for name, score in zip(recommendations['actor_names'], recommendations['scores']):
                    print(f"Actor: {name}, Score: {score:.4f}")
                
                # Get relevant actors for evaluation
                relevant_actors = user_actor_matrix.loc[user_id][user_actor_matrix.loc[user_id] > 0].index.tolist()
                print(f"\nRelevant actors for user {user_id}: {relevant_actors}")
                
            
            # Save model
            save_model(model)
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    if __name__ == "__main__":
        main()