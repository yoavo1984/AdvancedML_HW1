import numpy as np
import time

import data_loader
import shuffler
from Model.mf_model import  MFModel
from Model.hyper_parameters import *
from Learning.sgd_learner import SGDLearner
from Learning.als_learner import ALSLearner
from Evaluations.evaluations import *
import dataset

# POSSIBLE_D = [2,4,10,20,40,50,70,100,200]
POSSIBLE_D = [2,4,10,20,40]
MODEL_K = 10
RATINGS_SIZE = 5000
RECALL_K = 10

def deliverable_two(dataset, model):
    hyperparametersALS = MFALSHyperparameters(k=MODEL_K, alpha=0.002, gamma_array=[0.2, 0.2, 0.2, 0.2], epsilon=0.001)
    learner = ALSLearner()

    test_dataset = dataset.get_test_dataset()
    test_size = dataset.get_test_dataset_size()

    regularization = [0.1, 1, 10, 100, 1000]

    with open('deliverable/deliverable_2_data', 'w') as file:
        for value in regularization:
            model.reset_model()
            hyperparametersALS.gamma_array = [value] * 4

            learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS)

            metric_one = rmse(test_dataset['users'], model, test_size)
            _, metric_two = recall_k(test_dataset['users'], model, RECALL_K)

            file.write("{}::{}::{}\n".format(value, round(metric_one,2), round(metric_two,2)))

        print("-- Finished deliverable 2\n")


def deliverable_three(dataset, model):
    hyperparametersALS = MFALSHyperparameters(k=MODEL_K, alpha=0.002, gamma_array=[10]*4, epsilon=0.001)
    learner = ALSLearner()

    test_dataset = dataset.get_test_dataset()
    test_size = dataset.get_test_dataset_size()

    with open('deliverable/deliverable_3_data', 'w') as file:
        for d in POSSIBLE_D:
            model.k = d
            hyperparametersALS.k = d

            model.reset_model()

            learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS)

            metric_one = rmse(test_dataset['users'], model, test_size)
            _, metric_two = recall_k(test_dataset['users'], model, RECALL_K)

            file.write("{}::{}::{}\n".format(d, round(metric_one,2), round(metric_two,2)))

    print("-- Finished deliverable 3\n")

def deliverable_four(dataset, model):
    hyperparametersALS = MFALSHyperparameters(k=MODEL_K, alpha=0.002, gamma_array=[10]*4, epsilon=0.001)
    learner = ALSLearner()

    with open('deliverable/deliverable_4_data', 'w') as file:
        for d in POSSIBLE_D:
            model.k = d
            hyperparametersALS.k = d
            model.reset_model()

            # Time measurement.
            start = time.time()
            learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS)
            total_time = time.time() - start

            file.write("{}::{}\n".format(d, round(total_time,2)))

    print("-- Finished deliverable 4\n")


def deliverable_five(dataset, model):
    # Human readable
    human_readable_output(dataset, model, 1, h=5, history_flag=0)
    human_readable_output(dataset, model, 2, h=5, history_flag=0)
    human_readable_output(dataset, model, 3, h=5, history_flag=0)
    human_readable_output(dataset, model, 100, h=5, history_flag=0)
    human_readable_output(dataset, model, 200, h=5, history_flag=0)
    print("-- Finished deliverable 5\n")


def run_sgd(dataset, model):
    hyperparametersSGD = MFSGDHyperparameters(k=MODEL_K, alpha=0.02, gamma_array=[0.01, 0.01, 0.01, 0.01], epochs=10)

    learner = SGDLearner()
    learner.LearnModelFromDataUsingSGD(dataset, model, hyperparametersSGD)

    return model

def run_als(dataset, model):
    hyperparametersALS = MFALSHyperparameters(k=MODEL_K, alpha=0.002, gamma_array=[0.2, 0.2, 0.2, 0.2], epsilon=0.001)

    learner = ALSLearner()
    learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS)
    return model

def part_six():
    rating_dataset = dataset.Dataset("../data/movies.dat", "../data/ratings.dat", RATINGS_SIZE)

    train_dataset = rating_dataset.get_train_dataset()
    mu = dataset.Dataset.get_dataset_mean_rating(train_dataset)

    num_users = rating_dataset.get_number_of_users()
    num_movies = rating_dataset.get_number_of_movies()

    model = MFModel(num_users, num_movies, k=MODEL_K, mu=mu)

    trained_model = run_als(rating_dataset, model)


if __name__ == "__main__":
    rating_dataset = dataset.Dataset("../data/movies.dat", "../data/ratings.dat", RATINGS_SIZE)

    # Calculate mean.
    train_dataset = rating_dataset.get_train_dataset()
    mu = dataset.Dataset.get_dataset_mean_rating(train_dataset)

    num_users = rating_dataset.get_number_of_users()
    num_movies = rating_dataset.get_number_of_movies()

    model = MFModel(num_users, num_movies, k=MODEL_K, mu=mu)
    trained_model = run_als(rating_dataset, model)

    part_six()

    # deliverable_two(rating_dataset, model)
    # deliverable_three(rating_dataset, model)
    # deliverable_four(rating_dataset, model)
    # deliverable_five(rating_dataset, trained_model)

