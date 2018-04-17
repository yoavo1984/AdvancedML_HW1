import numpy as np

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
