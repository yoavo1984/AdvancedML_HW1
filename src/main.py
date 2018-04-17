import data_loader
import shuffler
from Model.mf_model import  MFModel
from Model.hyper_parameters import *
from Learning.sgd_learner import SGDLearner
from Learning.als_learner import ALSLearner
import dataset


def run_sgd(dataset, model):
    hyperparametersSGD = MFSGDHyperparameters(k=40, alpha=0.02, gamma_array=[0.5, 0.5, 0.5, 0.5], epochs=10)

    learner = SGDLearner()
    learner.LearnModelFromDataUsingSGD(dataset, model, hyperparametersSGD)


def run_als(dataset, model):
    hyperparametersALS = MFALSHyperparameters(k=40, alpha=0.02, gamma_array=[0.01, 0.01, 0.01, 0.01], epsilon=0.001)

    learner = ALSLearner()
    learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS)


if __name__ == "__main__":
    rating_dataset = dataset.Dataset("../data/movies.dat", "../data/ratings.dat")

    # Calculate mean.
    train_dataset = rating_dataset.get_train_dataset()
    mu = dataset.Dataset.get_dataset_mean_rating(train_dataset)

    num_users = rating_dataset.get_number_of_users()
    num_movies = rating_dataset.get_number_of_movies()

    model = MFModel(num_users, num_movies, k=5, mu=mu)

    run_als(train_dataset, model)
    # run_sgd(dataset, model)

