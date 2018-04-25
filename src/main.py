import time

import dataset
import utils.configuration_parser as configuration_parser

from Evaluations.evaluations import *
from Learning.als_learner import ALSLearner
from Learning.sgd_learner import SGDLearner
from Model.hyper_parameters import *
from Model.mf_model import MFModel

POSSIBLE_D = [2,4,10,20,40,50,70,100,200]
POSSIBLE_LAMBDA = [0.1, 1, 10, 100, 1000]
MODEL_D = 10
RATINGS_SIZE = -1
RECALL_K = 10

#=========================================== Delivrables ===============================================================
def deliverable_two(dataset, model):
    hyperparametersALS = MFALSHyperparameters(d=model.d, lambda_array=[0.2, 0.2, 0.2, 0.2], epsillon=0.001)
    learner = ALSLearner()

    test_dataset = dataset.get_test_dataset()
    test_size = dataset.get_test_dataset_size()

    with open('deliverable/deliverable_2_data', 'w') as file:
        for value in POSSIBLE_LAMBDA:
            model.reset_model()
            hyperparametersALS.lambda_array = [value] * 4

            learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS, False)

            metric_one = rmse(test_dataset['users'], model, test_size)
            _, metric_two = recall_k(test_dataset['users'], model, RECALL_K)

            file.write("{}::{}::{}\n".format(value, round(metric_one,2), round(metric_two,2)))

        print("- Finished deliverable 2\n")


def deliverable_three(dataset, model):
    hyperparametersALS = MFALSHyperparameters(d=model.d,lambda_array=[10] * 4, epsillon=0.001)
    learner = ALSLearner()

    test_dataset = dataset.get_test_dataset()
    test_size = dataset.get_test_dataset_size()

    with open('deliverable/deliverable_3_data', 'w') as file:
        for d in POSSIBLE_D:
            model.d = d
            hyperparametersALS.d = d

            model.reset_model()

            learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS, False)

            metric_one = rmse(test_dataset['users'], model, test_size)
            _, metric_two = recall_k(test_dataset['users'], model, RECALL_K)

            file.write("{}::{}::{}\n".format(d, round(metric_one,2), round(metric_two,2)))

    print("- Finished deliverable 3\n")

def deliverable_four(dataset, model):
    hyperparametersALS = MFALSHyperparameters(d=model.d, lambda_array=[10] * 4, epsillon=0.001)
    learner = ALSLearner()

    with open('deliverable/deliverable_4_data', 'w') as file:
        for d in POSSIBLE_D:
            model.d = d
            hyperparametersALS.d = d
            model.reset_model()

            # Time measurement.
            start = time.time()
            learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS, False)
            total_time = time.time() - start

            file.write("{}::{}\n".format(d, round(total_time,2)))

    print("- Finished deliverable 4\n")


def deliverable_five(dataset, model):
    # Human readable
    human_readable_output(dataset, model, 1, h=5, history_flag=0)
    human_readable_output(dataset, model, 2, h=5, history_flag=0)
    human_readable_output(dataset, model, 3, h=5, history_flag=0)
    human_readable_output(dataset, model, 34, h=5, history_flag=0)
    human_readable_output(dataset, model, 35, h=5, history_flag=0)
    print("-- Finished deliverable 5\n")


def run_als(dataset, model):
    hyperparametersALS = MFALSHyperparameters(d=model.d, lambda_array=[12, .5, 12, .5], epsillon=0.001)

    for d in range (14, 15):
        model.d = d
        hyperparametersALS.d = d
        model.reset_model()

        learner = ALSLearner()
        learner.LearnModelFromDataUsingALS(dataset, model, hyperparametersALS, False)
        print ("Rmse for {} is = {}".format(d, rmse(dataset.get_test_dataset()['users'], model)))

    return model


# =========================================== Part 6 main flow =========================================================
def run_part_six():
    print ("\nRunning part 6 - Main Flow\n"
           "--------------------------")
    # loading and splitting data
    rating_dataset = dataset.Dataset("../data/movies.dat", "../data/ratings.dat", RATINGS_SIZE)
    configuration_dict = configuration_parser.parse_configuration_file("configuration")

    train_dataset = rating_dataset.get_train_dataset()
    test_dataset = rating_dataset.get_test_dataset()

    mu = dataset.Dataset.get_dataset_mean_rating(train_dataset)
    num_users = rating_dataset.get_number_of_users()
    num_movies = rating_dataset.get_number_of_movies()

    print ("- Data loaded and split.")

    # Training model
    start = time.time()
    model = MFModel(num_users, num_movies, d=configuration_dict['d'], mu=mu)

    print("- Model and algorithm parameters set.")
    if configuration_dict['algorithm'] == "als":
        hyperparameters = MFALSHyperparameters(d=configuration_dict['d'],
                                               lambda_array=configuration_dict['lambda'],
                                               epsillon=configuration_dict['epsillon'])
        learner_six = ALSLearner()
        learner_six.LearnModelFromDataUsingALS(rating_dataset, model, hyperparameters)
    else :
        hyperparameters = MFSGDHyperparameters(d=configuration_dict['d'],
                                               lambda_array=configuration_dict['lambda'],
                                               epochs=configuration_dict['epochs'],
                                               alpha=configuration_dict['alpha'])
        learner_six = SGDLearner()
        learner_six.LearnModelFromDataUsingSGD(rating_dataset, model, hyperparameters)
    duration = time.time() - start

    print("- Learning over.")
    # compute metrices on test set
    file_output = get_part_six_metrices_output(test_dataset, model)

    print("- Metrics calculated.")
    # print to file (hyper, metrices, time of training)
    with open("output", "w") as file:
        file.write("Algorithm\n"
                   "---------\n"
                   "- {}\n\n".format(configuration_dict['algorithm'].upper()))

        file.write(str(hyperparameters))
        file.write(file_output)

        file.write("Running Time:\n"
                   "-------------\n"
                   "- {}".format(round(duration,4)))

    print("- Finished output to file\n")


# ============================================== Helpers ===============================================================
def build_model():
    rating_dataset = dataset.Dataset("../data/movies.dat", "../data/ratings.dat", RATINGS_SIZE)

    # Calculate mean.
    train_dataset = rating_dataset.get_train_dataset()
    mu = dataset.Dataset.get_dataset_mean_rating(train_dataset)

    num_users = rating_dataset.get_number_of_users()
    num_movies = rating_dataset.get_number_of_movies()

    model = MFModel(num_users, num_movies, d=MODEL_D, mu=mu)
    return model

def run_deliverables():
    rating_dataset = dataset.Dataset("../data/movies.dat", "../data/ratings.dat", RATINGS_SIZE)

    model = build_model()
    trained_model = run_als(rating_dataset, model)

    print("Running Deliverables\n"
          "--------------------\n")
    deliverable_two(rating_dataset, model)
    deliverable_three(rating_dataset, model)
    deliverable_four(rating_dataset, model)
    deliverable_five(rating_dataset, trained_model)

if __name__ == "__main__":
    run_part_six()
    # run_deliverables()

