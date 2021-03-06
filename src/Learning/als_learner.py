from Learning.learner import Learner
import numpy as np
from Evaluations.evaluations import *

TEST= "als_test"
TRAIN = "als_train"

class ALSLearner(Learner):
    def __init__(self):
        super().__init__()
        self.name = "ALS"

    def solve_b_m (self, dataset, model, hyperparameters):
        # sum (r_m,n) - (mu + b_n + u_m^t v_n) / lambda_b_m + countb_m

        # Enumerate over all users.
        for user_id, movies_ratings in dataset['users'].items():
            bias_sum = 0
            for rating in movies_ratings:
                movie_id, true_rating = *rating,
                bias_sum += true_rating - (model.mu + model.b_movie[movie_id-1] + model.u_v_dot(user_id, movie_id))

            count = len(movies_ratings)
            bias_sum = bias_sum / (count + hyperparameters.lambda_array[2])

            # Update user bias to the calculated sum.
            model.b_user[user_id-1] = bias_sum


    def solve_b_n (self, dataset, model, hyperparameters):
        # sum (r_m,n) - (mu + b_m + u_m^t v_n) / lambda_b_n + countb_n

        # Enumerate over all movies.
        for movie_id, users_ratings in dataset['movies'].items():
            bias_sum = 0
            for rating in users_ratings:
                user_id, true_rating = *rating,
                bias_sum += true_rating - (model.mu + model.b_user[user_id - 1] + model.u_v_dot(user_id, movie_id))

            count = len(users_ratings)
            bias_sum = bias_sum / (count + hyperparameters.lambda_array[1])

            # Update movie bias to the calculated sum.
            model.b_movie[movie_id-1] = bias_sum


    def solve_u_m (self, dataset, model, hyperparameters):
        # Enumerate over all users.
        for user_id, movies_ratings in dataset['users'].items():
            inverse_matrix = np.zeros((hyperparameters.d, hyperparameters.d))
            predicton_delta_vector = np.zeros((hyperparameters.d, 1))

            for rating in movies_ratings:
                movie_id, true_rating = *rating,
                inverse_matrix += model.v_v_outer(movie_id)

                predicton_delta_vector += (true_rating - (model.mu + model.b_movie[movie_id-1] +model.b_user[user_id - 1])) *\
                                          model.v[movie_id-1].reshape(hyperparameters.d, 1)


            inv = np.linalg.inv(inverse_matrix + np.eye(hyperparameters.d)*hyperparameters.lambda_array[0])

            # Update user latent vector.
            model.u[user_id-1] = np.dot(inv, predicton_delta_vector).T


    def solve_v_n (self, dataset, model, hyperparameters):
        for movie_id, users_ratings in dataset['movies'].items():
            inverse_matrix = np.zeros((hyperparameters.d, hyperparameters.d))
            predicton_delta_vector = np.zeros((hyperparameters.d, 1))

            for rating in users_ratings:
                user_id, true_rating = *rating,
                inverse_matrix += model.u_u_outer(user_id)

                predicton_delta_vector +=  (true_rating - (model.mu + model.b_movie[movie_id - 1] + model.b_user[user_id - 1])) * \
                                          model.u[user_id - 1].reshape(hyperparameters.d, 1)

            inv = np.linalg.inv(inverse_matrix + np.eye(hyperparameters.d)*hyperparameters.lambda_array[0])

            # Update movie latent vector.
            model.v[movie_id - 1] = np.dot(inv, predicton_delta_vector).T

    def ALSIteration(self, dataset, model, hyperparameters):
        self.solve_b_m(dataset, model, hyperparameters)
        self.solve_b_n(dataset, model, hyperparameters)
        self.solve_u_m(dataset, model, hyperparameters)
        self.solve_v_n(dataset, model, hyperparameters)

    def LearnModelFromDataUsingALS(self, dataset, model, hyperparameters, save_to_file=False):
        train_dataset = dataset.get_train_dataset()
        test_dataset = dataset.get_test_dataset()

        train_curr_loss = Learner.loss_function(train_dataset["users"], model, hyperparameters)
        test_curr_loss = Learner.loss_function(test_dataset["users"], model, hyperparameters)
        prev_loss = np.inf
        iterations = 0

        self.open_log_file(TRAIN, model, hyperparameters)
        self.open_log_file(TEST, model, hyperparameters)

        self.write_iteration_error_to_file(TRAIN, iterations, train_curr_loss)
        self.write_iteration_error_to_file(TEST, iterations, test_curr_loss)

        for iterations in range(1, 10):
            model.generate_prediction_matrix()
            self.ALSIteration(train_dataset, model, hyperparameters)

            train_curr_loss = Learner.loss_function(train_dataset['users'], model, hyperparameters)
            test_curr_loss = Learner.loss_function(test_dataset['users'], model, hyperparameters)
            prev_loss = train_curr_loss

            self.write_iteration_error_to_file(TRAIN, iterations, train_curr_loss)
            self.write_iteration_error_to_file(TEST, iterations, test_curr_loss)

