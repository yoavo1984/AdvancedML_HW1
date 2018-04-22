import numpy as np
from Learning.learner import Learner
import warnings

np.seterr(all='raise')

U_INDEX = 0
V_INDEX = 1
B_USER_INDEX = 2
B_MOVIE_INDEX = 3


class SGDLearner(Learner):
    def __init__(self):
        super().__init__()
        self.name = "SGD"

    def LearnModelFromDataUsingSGD(self, dataset, model, hyperparameters):
        alpha = hyperparameters.alpha
        gamma_array = hyperparameters.gamma_array
        train_dataset = dataset.get_train_dataset()
        test_dataset = dataset.get_test_dataset()

        datapoints = []

        for user_id in train_dataset["users"]:
            for movie_rate in train_dataset["users"][user_id]:
                movie_id = movie_rate[0]
                movie_true_rating = movie_rate[1]
                datapoints.append((user_id, movie_id, movie_true_rating))


        curr_loss = Learner.loss_function(train_dataset["users"], model, hyperparameters)
        prev_loss = np.inf
        iterations = 0

        size_of_data={}
        size_of_data["train"] = dataset.calculate_size_of_data_set(train_dataset["users"])
        size_of_data["test"] = dataset.calculate_size_of_data_set(test_dataset["users"])

        self.open_log_file(model, hyperparameters)
        self.write_iteration_error_to_file(iterations, curr_loss)

        for epoch in range(hyperparameters.epochs):
            np.random.shuffle(datapoints)
            alpha = alpha / 5
            for point in datapoints:
                user_id, movie_id, movie_true_rating = *point,
                prediction = model.predict(user_id, movie_id)
                error = movie_true_rating - prediction
                error = round(float(error), 2)
                if error > 1000 or error < -1000:
                    pass

                try:
                    model.u[user_id - 1] = model.u[user_id - 1] + alpha * (
                        error * model.v[movie_id - 1] + gamma_array[U_INDEX] * model.u[user_id - 1])
                except FloatingPointError as e:
                    print(e)
                    print("hie")

                try:
                    model.v[movie_id - 1] = model.v[movie_id - 1] + alpha * (
                        error * model.u[user_id - 1] + gamma_array[V_INDEX] * model.v[movie_id - 1])
                except FloatingPointError as e:
                    print(e)
                    print("hie")

                try:
                    model.b_user[user_id - 1] = model.b_user[user_id - 1] + alpha * (
                        error + gamma_array[B_USER_INDEX] * model.b_user[user_id - 1])
                except FloatingPointError as e:
                    print(e)
                    print("hie")

                try:
                    model.b_movie[movie_id - 1] = model.b_movie[movie_id - 1] + alpha * (
                        error + gamma_array[B_MOVIE_INDEX] * model.b_movie[movie_id - 1])
                except FloatingPointError as e:
                    print (e)
                    print("hie")

            print("###########")
            print(Learner.loss_function(train_dataset["users"], model, hyperparameters))

            if test_dataset:
                print(Learner.loss_function(test_dataset["users"], model, hyperparameters))
