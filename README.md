# RECOMMENDATION-SYSTEM-

COMPANY: CODETECH IT SOLUTIONS

NAME: VADLAMUDI PRASANNA

INTERN ID: CT08DF213

DOMAIN: MACHINE LEARNING

DURATION: 8 WEEKS

MENTOR: NEELA SANTOSH

##This project implements a movie recommendation system using collaborative filtering through matrix factorization with PyTorch. It is trained on the MovieLens 100K dataset, which contains 100,000 ratings from 943 users on 1,682 movies. The goal is to predict how a user might rate a movie they haven't yet watched and recommend the highest-rated unseen movies.

The core idea is matrix factorization, where both users and movies are represented as vectors (embeddings) in a shared latent space. The predicted rating is calculated as the dot product of the user and item vectors, adjusted by user and item bias terms. These embeddings and biases are learned during training by minimizing the Mean Squared Error (MSE) between predicted ratings and actual ratings.

The data is first loaded and preprocessed using pandas. User IDs and item (movie) IDs are converted to internal integer indices to be used with PyTorch’s embedding layers. The dataset is then split into training and test sets using scikit-learn’s train_test_split function.

A custom PyTorch model called MatrixFactorization is defined. It includes embedding layers for users and items, along with user and item biases. The model is trained using the Adam optimizer over 20 epochs, optimizing the MSE loss. After training, the model is evaluated using RMSE (Root Mean Squared Error) to determine how well it generalizes to unseen data.

The recommendation function takes a user ID and predicts ratings for all movies, returning the top N recommendations based on the predicted scores. These recommendations are made for movies the user has not already rated.

The code demonstrates the effectiveness of matrix factorization for collaborative filtering and provides a base for more advanced recommendation systems. Possible extensions include incorporating movie metadata (such as titles or genres), using neural networks for deeper interaction modeling, or deploying the system as a web application using frameworks like Flask or Streamlit.

Overall, this project combines PyTorch, pandas, and scikit-learn to create a functional recommendation engine that can be adapted and extended for real-world applications.

OUTPUT: 
<img width="561" height="745" alt="Image" src="https://github.com/user-attachments/assets/1a67f09e-2e8c-4b6d-b0c1-99153b6b9d47" />
