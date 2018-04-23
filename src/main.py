import numpy as np

import data_loader
import shuffler
from Model.mf_model import  MFModel
from Model.hyper_parameters import *
from Learning.sgd_learner import SGDLearner
from Learning.als_learner import ALSLearner
import dataset

MODEL_K = 20
RATINGS_SIZE = 100000

def deliverable_two(dataset, model):
    hyperparametersALS = MFALSHyperparameters(k=MODEL_K, alpha=0.002, gamma_array=[0.2, 0.2, 0.2, 0.2], epsilon=0.001)

    regularization = [0.1, 1, 10, 100, 1000]
    for value in regularization:
        hyperparametersALS.gamma_array = [value] * 4
        learner = ALSLearner()
        learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS)


def run_sgd(dataset, model):
    hyperparametersSGD = MFSGDHyperparameters(k=MODEL_K, alpha=0.02, gamma_array=[0.01, 0.01, 0.01, 0.01], epochs=10)

    learner = SGDLearner()
    learner.LearnModelFromDataUsingSGD(dataset, model, hyperparametersSGD)


def run_als(dataset, model):
    hyperparametersALS = MFALSHyperparameters(k=MODEL_K, alpha=0.002, gamma_array=[0.2, 0.2, 0.2, 0.2], epsilon=0.001)

    learner = ALSLearner()
    learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS)

if __name__ == "__main__":
    rating_dataset = dataset.Dataset("../data/movies.dat", "../data/ratings.dat", RATINGS_SIZE)

    # Calculate mean.
    train_dataset = rating_dataset.get_train_dataset()
    mu = dataset.Dataset.get_dataset_mean_rating(train_dataset)

    num_users = rating_dataset.get_number_of_users()
    num_movies = rating_dataset.get_number_of_movies()

    model = MFModel(num_users, num_movies, k=MODEL_K, mu=mu)

    run_als(rating_dataset, model)
    # run_sgd(rating_dataset, model)
    # deliverable_two(rating_dataset, model)

