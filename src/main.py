import data_loader
import shuffler
from Model.mf_model import  MFModel
from Model.hyper_parameters import *
from Learning.sgd_learner import SGDLearner
from Learning.als_learner import ALSLearner


def run_sgd(dataset, model):
    hyperparametersSGD = MFSGDHyperparameters(k=5, alpha=0.02, gamma_array=[0.5, 0.5, 0.5, 0.5], epochs=10)

    learner = SGDLearner()
    learner.LearnModelFromDataUsingSGD(dataset, model, hyperparametersSGD)

def run_als(dataset, model):
    hyperparametersALS = MFALSHyperparameters(k=5, alpha=0.02, gamma_array=[0.5, 0.5, 0.5, 0.5], epsilon=0.001)

    learner = ALSLearner()
    learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS)

if __name__ == "__main__":
    dataset = {}

    # Generating users datasets.
    dataset["users"] = data_loader.generate_users_dict("../data/ratings.dat")
    dataset["users_train"], dataset["users_test"] = shuffler.split_data_randomly(dataset["users"])

    # Generating movies datasets.
    dataset["movies_train"] = data_loader.generate_movies_dict_from_users_dict(dataset["users_train"])
    dataset["movies_test"] = data_loader.generate_movies_dict_from_users_dict(dataset["users_test"])

    # Calculate mean.
    mu = data_loader.calculate_dataset_mu(dataset["users_train"])

    model = MFModel(len(dataset["users_train"]), 4000, k=5, mu=mu)

    run_als(dataset, model)
    # run_sgd(dataset, model)

