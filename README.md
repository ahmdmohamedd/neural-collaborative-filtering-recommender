# Neural Collaborative Filtering Recommender

This repository contains the implementation of a Neural Collaborative Filtering (NCF) model for building a movie recommendation system. The system predicts user ratings for movies based on historical interactions using deep learning. The MovieLens dataset is used to train and evaluate the model.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)

## Overview

This recommendation system leverages **Neural Collaborative Filtering (NCF)**, a deep learning technique that combines matrix factorization and multi-layer perceptrons (MLP). The model predicts a user's rating for a movie and aims to provide personalized recommendations. The core idea behind NCF is to learn embeddings for users and movies, then use these embeddings to predict ratings.

### Key Features:
- User and movie embeddings are learned via neural networks.
- The model is trained using the **MovieLens dataset**.
- Evaluation is performed using **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)** metrics.

## Installation

To run this project locally, follow the instructions below:

### Prerequisites:
- Python 3.x
- PyTorch
- Pandas
- Numpy
- Scikit-learn

### Install Dependencies

Clone this repository:

```bash
git clone https://github.com/ahmdmohamedd/neural-collaborative-filtering-recommender.git
cd neural-collaborative-filtering-recommender
```

Create a virtual environment (optional, but recommended):

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

Install required packages:

```bash
pip install -r requirements.txt
```

## Dataset

The model uses the **MovieLens dataset**, a collection of movie ratings data, which can be downloaded from [here](https://grouplens.org/datasets/movielens/). We use the "ml-latest-small" version of the dataset, which contains user ratings for movies.

The dataset is expected to have two primary CSV files:
- `movies.csv` (movie information)
- `ratings.csv` (user ratings)

### Data Preprocessing:
- **User and movie IDs** are encoded using **LabelEncoder** to convert them into numerical values suitable for embedding layers.
- **Rating** values are used as the target variable for training the model.

## Usage

1. **Training the Model**:  
   Run the `ncf_recommender.ipynb` Jupyter notebook to train the model. The notebook covers data preprocessing, model definition, training, and evaluation.

2. **Evaluation**:  
   After training, the model’s performance is evaluated using **RMSE** and **MAE** to assess its accuracy on the validation set.

3. **Make Predictions**:  
   Once the model is trained, you can make predictions on new user-movie pairs to recommend movies based on the learned embeddings.

## Model Architecture

The model uses a **Neural Collaborative Filtering (NCF)** architecture, which includes:

1. **Embedding Layers**:  
   - **User Embeddings**: A dense vector representation of each user.
   - **Movie Embeddings**: A dense vector representation of each movie.

2. **Multi-Layer Perceptron (MLP)**:  
   - The embeddings are concatenated and passed through a feed-forward neural network (MLP) with one hidden layer. The model is trained to predict the user's rating for a movie.

### Hyperparameters:
- **Embedding Dimension**: 50
- **Hidden Layer Size**: 128
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)

## Evaluation Metrics

We evaluate the model using the following metrics:

- **Root Mean Squared Error (RMSE)**: Measures the square root of the average squared differences between predicted and actual ratings. RMSE penalizes large errors.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual ratings. MAE is more interpretable and less sensitive to large errors.

## Results

The model was trained on the MovieLens dataset and evaluated on a validation set. The evaluation metrics achieved are:

- **Validation RMSE**: 0.9326
- **Validation MAE**: 0.7154

These results indicate that the model performs well, with average errors of approximately 0.93 for RMSE and 0.72 for MAE.

## Contributing

Contributions to this repository are welcome. If you have suggestions, improvements, or bug fixes, feel free to create a pull request.

### Steps to contribute:
1. Fork this repository.
2. Clone your fork: `git clone https://github.com/your-username/neural-collaborative-filtering-recommender.git`
3. Create a new branch: `git checkout -b feature-branch`
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-branch`
6. Submit a pull request.
