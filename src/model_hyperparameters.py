import numpy as np
import data_loader
import shuffler
import ALSSolver


class MFHyperparameters:
    def __init__(self, k, alpha, gamma_array):
        """
        Bn Bias
        U users matrix
        V items matrix
        m users index
        n items index
        """

        self.gamma_array = gamma_array
        self.k = k
        self.alpha = alpha


class MFALSHyperparameters(MFHyperparameters):
    def __init__(self, k, alpha, gamma_array, epsilon):
        super(MFALSHyperparameters, self).__init__(k, alpha, gamma_array)
        self.epsilon = epsilon


class MFSGDHyperparameters(MFHyperparameters):
    def __init__(self, k, alpha, gamma_array, epochs):
        super(MFSGDHyperparameters, self).__init__(k, alpha, gamma_array)
        self.epochs = epochs


U_INDEX = 0
V_INDEX = 1
B_USER_INDEX = 2
B_MOVIE_INDEX = 3


class MFModel():
    def __init__(self, num_users, num_items, k, mu):
        self.u = np.random.normal(scale=0.25, size=(num_users, k)).round(2)
        # self.u = self.u.view(np.float128)

        self.v = np.random.normal(scale=0.25, size=(num_items, k)).round(2)
        # self.v = self.v.view(np.float128)

        self.b_user = np.random.normal(scale=0.25, size=num_users).round(2)
        # self.b_user = self.b_user.view(np.float128)

        self.b_movie = np.random.normal(scale=0.25, size=num_items).round(2)
        # self.b_movie = self.b_movie.view(np.float128)

        self.mu = mu

    def predict(self, user_id, movie_id):
        result = self.mu + np.dot(self.u[user_id - 1], self.v[movie_id - 1]) + self.b_user[user_id - 1] + self.b_movie[
            movie_id - 1]
        return round(float(result), 2)

    def u_v_dot(self, user_id, movie_id):
        return np.dot(self.u[user_id-1], self.v[movie_id-1])

    def u_u_outer(self, user_id):
        return np.outer(self.u[user_id-1], self.u[user_id-1])

    def v_v_outer(self, movie_id):
        return np.outer(self.v[movie_id - 1], self.v[movie_id - 1])



def calculate_dataset_mu(dataset):
    sum_of_rates = 0
    num_of_rates = 0
    for key in dataset:
        for movie_rate in dataset[key]:
            rate = movie_rate[1]
            sum_of_rates += rate
            num_of_rates += 1
    return round(float(sum_of_rates / num_of_rates), 2)


def loss_function(dataset, model, hyperparameters):
    sum = 0
    for user_id in dataset:
        for movie_rate in dataset[user_id]:
            movie_id = movie_rate[0]
            movie_true_rating = movie_rate[1]
            prediction = model.predict(user_id, movie_id)
            sum += pow(movie_true_rating - prediction, 2)

    sum = sum / 2

    sum += (hyperparameters.gamma_array[U_INDEX] / 2) * np.linalg.norm(model.v) + \
           (hyperparameters.gamma_array[V_INDEX] / 2) * np.linalg.norm(model.u) + \
           (hyperparameters.gamma_array[B_USER_INDEX] / 2) * np.linalg.norm(model.b_user) + \
           (hyperparameters.gamma_array[B_MOVIE_INDEX] / 2) * np.linalg.norm(model.b_movie)

    return sum


def LearnModelFromDataUsingSGD(dataset, model, hyperparameters):
    alpha = hyperparameters.alpha
    gamma_array = hyperparameters.gamma_array

    datapoints = []

    for user_id in dataset["users_train"]:
        for movie_rate in dataset["users_train"][user_id]:
            movie_id = movie_rate[0]
            movie_true_rating = movie_rate[1]
            datapoints.append((user_id, movie_id, movie_true_rating))

    print(loss_function(dataset["users_train"], model, hyperparameters))
    print(loss_function(dataset["users_test"], model, hyperparameters))

    for epoch in range(hyperparameters.epochs):
        np.random.shuffle(datapoints)
        # u = model.u.copy()
        # v = model.v.copy()
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
        print(loss_function(dataset["users_train"], model, hyperparameters))
        print(loss_function(dataset["users_test"], model, hyperparameters))


def LearnModelFromDataUsingALS(dataset, model, hyperparameters):
    prev_loss = np.inf
    curr_loss = loss_function(dataset['users_train'], model, hyperparameters)
    print(curr_loss)

    for i in range(10):

        ALSSolver.solve_b_m(dataset, model, hyperparameters)
        ALSSolver.solve_b_n(dataset, model, hyperparameters)
        ALSSolver.solve_u_m(dataset, model, hyperparameters)
        ALSSolver.solve_v_n(dataset, model, hyperparameters)

        print("##### {} #####".format(i))
        prev_loss = curr_loss
        curr_loss = loss_function(dataset['users_train'], model, hyperparameters)
        print(curr_loss)


if __name__ == "__main__":
    dataset = {}
    dataset["users"] = data_loader.generate_users_dict("../data/ratings.dat")
    dataset["users_train"], dataset["users_test"] = shuffler.split_data_randomly(dataset["users"])
    dataset["movies_train"] = data_loader.generate_movies_dict_from_users_dict(dataset["users_train"])
    dataset["movies_test"] = data_loader.generate_movies_dict_from_users_dict(dataset["users_test"])

    mu = calculate_dataset_mu(dataset["users_train"])

    hyperparametersSGD = MFSGDHyperparameters(k=5, alpha=0.02, gamma_array=[0.5, 0.5, 0.5, 0.5], epochs=10)
    hyperparametersALS = MFALSHyperparameters(k=5, alpha=0.02, gamma_array=[0.5, 0.5, 0.5, 0.5], epsilon=0.001)
    model = MFModel(len(dataset["users_train"]), 4000, k=5, mu=mu)

    #LearnModelFromDataUsingSGD(dataset, model, hyperparametersSGD)
    LearnModelFromDataUsingALS(dataset, model, hyperparametersALS)