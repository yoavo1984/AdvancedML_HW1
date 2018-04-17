import data_loader
import shuffler
from Model.mf_model import  MFModel
from Model.hyper_parameters import *
from Learning.sgd_learner import SGDLearner
from Learning.als_learner import ALSLearner
from Evaluations.evaluations import *


def run_sgd(dataset, model):
    hyperparametersSGD = MFSGDHyperparameters(k=40, alpha=0.02, gamma_array=[0.5, 0.5, 0.5, 0.5], epochs=10)

    learner = SGDLearner()
    learner.LearnModelFromDataUsingSGD(dataset, model, hyperparametersSGD)


def run_als(dataset, model):
    hyperparametersALS = MFALSHyperparameters(k=40, alpha=0.02, gamma_array=[0.01, 0.01, 0.01, 0.01], epsilon=0.001)

    learner = ALSLearner()
    learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS)


if __name__ == "__main__":
    dataset = {}

    # Generating users datasets.
    dataset["users"] = data_loader.generate_users_dict("../data/ratings.dat", num_ratings=500000)
    dataset["users_train"], dataset["users_test"] = shuffler.split_data_randomly(dataset["users"])

    # Generating movies datasets.
    dataset["movies_train"] = data_loader.generate_movies_dict_from_users_dict(dataset["users_train"])
    dataset["movies_test"] = data_loader.generate_movies_dict_from_users_dict(dataset["users_test"])

    # Calculate mean.
    mu = data_loader.calculate_dataset_mu(dataset["users_train"])
    # calculate size of data
    # size_of_data = data_loader.calculate_size_of_data_set(dataset["users"])

    model = MFModel(len(dataset["users_train"]), 3960, k=40, mu=mu)

    run_als(dataset, model)


    # run_sgd(dataset, model)

