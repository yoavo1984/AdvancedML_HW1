import numpy as np
from Learning.learner import Learner

U_INDEX = 0
V_INDEX = 1
B_USER_INDEX = 2
B_MOVIE_INDEX = 3


class SGDLearner(Learner):
    def __init__(self):
        super().__init__()
        self.name = "SGD"

    def LearnModelFromDataUsingSGD(self, dataset, model, hyperparameters, test_dataset):
        alpha = hyperparameters.alpha
        gamma_array = hyperparameters.gamma_array

        datapoints = []

        for user_id in dataset["users_train"]:
            for movie_rate in dataset["users_train"][user_id]:
                movie_id = movie_rate[0]
                movie_true_rating = movie_rate[1]
                datapoints.append((user_id, movie_id, movie_true_rating))

        print(Learner.loss_function(dataset["users_train"], model, hyperparameters))
        print(Learner.loss_function(dataset["users_test"], model, hyperparameters))

        for epoch in range(hyperparameters.epochs):
            np.random.shuffle(datapoints)

            for point in datapoints:
                user_id, movie_id, movie_true_rating = *point,
                prediction = model.predict(user_id, movie_id)
                error = movie_true_rating - prediction
                error = round(float(error), 2)

                model.u[user_id - 1] = model.u[user_id - 1] + alpha * (
                    error * model.v[movie_id - 1] + gamma_array[U_INDEX] * model.u[user_id - 1])
                model.v[movie_id - 1] = model.v[movie_id - 1] + alpha * (
                    error * model.u[user_id - 1] + gamma_array[V_INDEX] * model.v[movie_id - 1])

                model.b_user[user_id - 1] = model.b_user[user_id - 1] + alpha * (
                    error + gamma_array[B_USER_INDEX] * model.b_user[user_id - 1])
                model.b_movie[movie_id - 1] = model.b_movie[movie_id - 1] + alpha * (
                    error + gamma_array[B_MOVIE_INDEX] * model.b_movie[movie_id - 1])

            print("###########")
            print(Learner.loss_function(dataset["users_train"], model, hyperparameters))

            if test_dataset:
                print(Learner.loss_function(dataset["users_test"], model, hyperparameters))
