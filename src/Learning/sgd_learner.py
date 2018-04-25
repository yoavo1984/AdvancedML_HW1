import numpy as np
from Learning.learner import Learner
import warnings

np.seterr(all='raise')

U_INDEX = 0
V_INDEX = 1
B_USER_INDEX = 2
B_MOVIE_INDEX = 3

TEST= "sgd_test"
TRAIN = "sgd_train"

class SGDLearner(Learner):
    def __init__(self):
        super().__init__()
        self.name = "SGD"

    def LearnModelFromDataUsingSGD(self, dataset, model, hyperparameters, save_to_file=False):
        alpha = hyperparameters.alpha
        lambda_array = hyperparameters.lambda_array
        train_dataset = dataset.get_train_dataset()
        test_dataset = dataset.get_test_dataset()

        datapoints = []

        for user_id in train_dataset["users"]:
            for movie_rate in train_dataset["users"][user_id]:
                movie_id = movie_rate[0]
                movie_true_rating = movie_rate[1]
                datapoints.append((user_id, movie_id, movie_true_rating))


        train_loss = Learner.loss_function(train_dataset["users"], model, hyperparameters)
        test_loss = Learner.loss_function(test_dataset["users"], model, hyperparameters)
        iterations = 0

        self.open_log_file(TRAIN, model, hyperparameters)
        self.open_log_file(TEST, model, hyperparameters)

        self.write_iteration_error_to_file(TRAIN, iterations, train_loss)
        self.write_iteration_error_to_file(TEST, iterations, test_loss)

        for epoch in range(hyperparameters.epochs):
            np.random.shuffle(datapoints)
            alpha = alpha / 3
            for point in datapoints:
                user_id, movie_id, movie_true_rating = *point,
                prediction = model.predict(user_id, movie_id)
                error = movie_true_rating - prediction

                model.u[user_id - 1] = model.u[user_id - 1] + alpha * (
                    error * model.v[movie_id - 1] + lambda_array[U_INDEX] * model.u[user_id - 1])

                model.v[movie_id - 1] = model.v[movie_id - 1] + alpha * (
                    error * model.u[user_id - 1] + lambda_array[V_INDEX] * model.v[movie_id - 1])

                model.b_user[user_id - 1] = model.b_user[user_id - 1] + alpha * (
                    error + lambda_array[B_USER_INDEX] * model.b_user[user_id - 1])

                model.b_movie[movie_id - 1] = model.b_movie[movie_id - 1] + alpha * (
                    error + lambda_array[B_MOVIE_INDEX] * model.b_movie[movie_id - 1])

            iterations += 1
            train_loss = Learner.loss_function(train_dataset["users"], model, hyperparameters)
            test_loss = Learner.loss_function(test_dataset["users"], model, hyperparameters)
            self.write_iteration_error_to_file(TRAIN, iterations, train_loss)
            self.write_iteration_error_to_file(TEST, iterations, test_loss)