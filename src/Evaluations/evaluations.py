import data_loader
import math
import numpy as np
import time



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

def se(dataset, model):
    sum = 0
    for user_id in dataset:
        for movie_rate in dataset[user_id]:
            movie_id = movie_rate[0]
            movie_true_rating = movie_rate[1]
            prediction = model.predict(user_id, movie_id)
            sum += pow(movie_true_rating - prediction, 2)

    return sum


def rmse(dataset, model, size_of_data):
    # calculate squared error
    sum = 0
    # need to compute one time and reuse

    for user_id in dataset:
        for movie_rate in dataset[user_id]:
            movie_id = movie_rate[0]
            movie_true_rating = movie_rate[1]
            prediction = model.predict(user_id, movie_id)
            sum += pow(movie_true_rating - prediction, 2)
    # take the mean
    sum = sum / size_of_data

    # take the root
    sum = math.sqrt(sum)

    return sum

def mean_percentile_rank(dataset_users, model, k):
    """
    MPR = sum(true rank*predicted rank percentile) / sum of true ranks
    """
    users_k=np.zeros(7000)
    for user_id in dataset_users:
        predictions = model.get_user_predictions(user_id)
        indexes = np.argsort(predictions)
        percentiles = np.zeros(4000)
        for rank, movie_id in enumerate(indexes[::-1]):
            percentile = rank / len(indexes)
            percentiles[rank] = percentile

        ground_truth_set = create_ground_truth(dataset_users[user_id], len(dataset_users[user_id]))
        # for item in ground_truth_set:
        #     sum += item*





def create_ground_truth(user_data, k):
    sorted_by_rate = sorted(user_data, key=lambda tup: tup[1])
    only_movies = [movie[0] for movie in sorted_by_rate[-k::]]
    ground_set = set(only_movies)

    return ground_set


def create_predicted(dataset, model, user_id, k):
    predictions = model.get_user_predictions(user_id)
    predictions = np.argsort(predictions)
    only_movies = [movie for movie in predictions[-k::]]
    predicted_set = set(only_movies)
    return predicted_set

def precision_k(dataset_users, dataset_movies, model, k):
    """
    TP = # ground truth intersect predict (top k)
    FP = # results in top k but not in ground truth
    p@k = TP / (TP+FP) = TP / k
    """
    users_k=np.zeros(7000)
    for user_id in dataset_users:
        # print (user_id)
        ground_truth_set = create_ground_truth(dataset_users[user_id], k)
        # maybe iterate over all movies? keep data of all movies??
        predicted_set = create_predicted(dataset_movies, model, user_id, k)
        # because set.intersection would return ground truth after removing predicted from it
        tp = ground_truth_set.intersection(predicted_set)
        fp = predicted_set.difference(ground_truth_set)

        # result = len(tp) / (len(tp) + len(fp))
        # k-1???
        result = len(tp) / k
        # if result != result1:
        #     print ("oy")
        users_k[user_id-1] = result

    return np.mean(users_k)


def recall_k(dataset_users, dataset_movies, model, k):
    """
    TP = # ground truth intersect predict (top k)
    TN = # results in ground truth but not in top k
    N = total # of results in ground truth
    R@k = TP / (TP+TN) = TP / N
    """
    users_k=np.zeros(7000)
    for user_id in dataset_users:
        ground_truth_set = create_ground_truth(dataset_users[user_id], k)
        predicted_set = create_predicted(dataset_movies, model, user_id, k)
        # because set.intersection would return predicted after removing ground truth from it
        tp = predicted_set.intersection(ground_truth_set)
        tn = ground_truth_set.difference(predicted_set)

        result = len(tp) / (len(tp) + len(tn))
        # n???
        # result1 = len(tp) / len(dataset_users[user_id])
        # if result != result1:
        #     print ("oy")
        users_k[user_id-1] = result

    return np.mean(users_k)


def map(self, dataset, model):
    pass


def run_metrices(dataset, model, k, size_of_data):
    # flag 0/1 is to prevent calculation of size of train and test... not good implementation
    print ("training")
    mpr = mean_percentile_rank(dataset["users_train"], model, k)
    se_score = se(dataset["users_train"], model)
    rmse_score = rmse(dataset["users_train"], model, size_of_data["train"])
    prec_k = precision_k(dataset["users_train"],dataset["movies_train"] , model, k)
    reca_k = recall_k(dataset["users_train"],dataset["movies_train"], model, k)
    print (mpr, se_score, rmse_score, prec_k, reca_k)

    print ("test")
    mpr = mean_percentile_rank(dataset["users_train"], model, k)
    se_score = se(dataset["users_test"], model)
    rmse_score = rmse(dataset["users_test"], model, size_of_data["test"])
    prec_k = precision_k(dataset["users_test"], dataset["movies_test"] , model, k)
    reca_k = recall_k(dataset["users_test"], dataset["movies_test"], model, k)
    print (mpr, se_score, rmse_score, prec_k, reca_k)


if __name__ == "__main__":
    pass