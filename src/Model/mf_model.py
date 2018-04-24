import numpy as np

class MFModel():
    def __init__(self, num_users, num_items, d, mu):
        # Saving model parameters.
        self.num_users = num_users
        self.num_items = num_items
        self.d = d
        self.mu = mu

        # Initializing object variables.
        self.u = []
        self.v = []
        self.b_user = []
        self.b_movie = []
        self.prediction_matrix = []

        # Set initial values.
        self.reset_model()

    def reset_model(self):
        self.u = np.random.normal(scale=0.25, size=(self.num_users, self.d)).round(2)

        self.v = np.random.normal(scale=0.25, size=(self.num_items, self.d)).round(2)

        self.b_user = np.random.normal(scale=0.25, size=self.num_users).round(2)

        self.b_movie = np.random.normal(scale=0.25, size=self.num_items).round(2)

    def predict(self, user_id, movie_id):
        result = self.mu + np.dot(self.u[user_id - 1], self.v[movie_id - 1]) + self.b_user[user_id - 1] + self.b_movie[movie_id - 1]
        return round(float(result), 2)

    def get_user_predictions(self, user_id):
        return self.prediction_matrix[user_id-1]

    def get_movie_predictions(self, movie_id):
        return self.prediction_matrix[movie_id-1]

    def generate_prediction_matrix(self):
        # maybe mistake is here????
        self.prediction_matrix = np.dot(self.u, self.v.T) + self.b_movie.T + self.b_user.reshape(len(self.b_user),1) + self.mu

    def u_v_dot(self, user_id, movie_id):
        return np.dot(self.u[user_id-1], self.v[movie_id-1])

    def u_u_outer(self, user_id):
        return np.outer(self.u[user_id-1], self.u[user_id-1])

    def v_v_outer(self, movie_id):
        return np.outer(self.v[movie_id - 1], self.v[movie_id - 1])
