# Imports for testing.
from  Model.hyper_parameters import MFHyperparameters

# real imorts
import numpy as np
DEBUG = False

U_INDEX = 0
V_INDEX = 1
B_USER_INDEX = 2
B_MOVIE_INDEX = 3

class Learner(object):

    def __init__(self):
        self.name = "Learner"
        self.log_file = {}

    @staticmethod
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

    def open_log_file(self ,log_name ,model, hyperparameters):
        file_name = "{}_k-{}_l-{}".format(log_name, hyperparameters.d, hyperparameters.gamma_array[0])
        self.log_file[log_name] = open(file_name, 'w')
        self.log_file[log_name].write("#"*10 +"\t" + file_name + "\t" +"#"*10 + "\n\n")

    def write_iteration_error_to_file(self, log_file, iteration, error):
        content = "{}::{}\n".format(iteration, error)
        self.log_file[log_file].write(content)

        if DEBUG:
            print(content)

    def close_log_file(self, log_file):
        self.log_file[log_file].close()


if __name__ == "__main__":
    mock_hyper = MFHyperparameters(k=4, alpha=0.1, lambda_array=[1]);
    learner = Learner()
    learner.open_log_file("model", mock_hyper)
    learner.write_error_to_file("iteration 1 : error 100000")
    learner.close_log_file()