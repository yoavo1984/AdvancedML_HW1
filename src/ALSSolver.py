import numpy as np

def compute_b_m (dataset, model, hyperparameters):
    # sum (r_m,n) - (mu + b_n + u_m^t v_n) / lambda_b_m + countb_m

    # Enumerate over all users.
    for user_id, movies_ratings in dataset['users_train'].items():
        bias_sum = 0
        for rating in movies_ratings:
            movie_id, true_rating = *rating,
            bias_sum += true_rating - (model.mu + model.b_movie[movie_id-1] + model.u_v_dot(user_id, movie_id))


        count = len(movies_ratings)
        bias_sum = bias_sum / (count + hyperparameters.gamma_array[2])

        # Update user bias to the calculated sum.
        model.b_user[user_id-1] = bias_sum


def compute_b_n (dataset, model, hyperparameters):
    # sum (r_m,n) - (mu + b_m + u_m^t v_n) / lambda_b_n + countb_n

    # Enumerate over all movies.
    for movie_id, users_ratings in dataset['movies_train'].items():
        bias_sum = 0
        for rating in users_ratings:
            user_id, true_rating = *rating,
            bias_sum += true_rating - (model.mu + model.b_user[user_id - 1] + model.u_v_dot(user_id, movie_id))

        count = len(users_ratings)
        bias_sum = bias_sum / (count + hyperparameters.gamma_array[1])

        # Update user bias to the calculated sum.
        model.b_movie[movie_id-1] = bias_sum


def compute_u_m (dataset, model, hyperparameters):
    # Enumerate over all users.
    for user_id, movies_ratings in dataset['users_train'].items():
        inverse_matrix = np.zeros((hyperparameters.k, hyperparameters.k))
        predicton_delta_vector = np.zeros((hyperparameters.k, 1))

        for rating in movies_ratings:
            movie_id, true_rating = *rating,
            inverse_matrix += model.v_v_outer(movie_id)

            predicton_delta_vector += (true_rating - (model.mu + model.b_movie[movie_id-1] +model.b_user[user_id - 1])) *\
                                      model.v[movie_id-1].reshape(hyperparameters.k, 1)


        inv = np.linalg.inv(inverse_matrix + np.eye(hyperparameters.k)*hyperparameters.gamma_array[0])

        # Update user latent vector.
        model.u[user_id-1] = np.dot(inv, predicton_delta_vector).T


def compute_v_n (dataset, model, hyperparameters):
    for movie_id, users_ratings in dataset['movies_train'].items():
        inverse_matrix = np.zeros((hyperparameters.k, hyperparameters.k))
        predicton_delta_vector = np.zeros((hyperparameters.k, 1))

        for rating in users_ratings:
            user_id, true_rating = *rating,
            inverse_matrix += model.u_u_outer(user_id)

            predicton_delta_vector +=  (true_rating - (model.mu + model.b_movie[movie_id - 1] + model.b_user[user_id - 1])) * \
                                      model.u[user_id - 1].reshape(hyperparameters.k, 1)

        inv = np.linalg.inv(inverse_matrix + np.eye(hyperparameters.k)*hyperparameters.gamma_array[0])

        # Update user latent vector.
        model.v[movie_id - 1] = np.dot(inv, predicton_delta_vector).T