import math
import numpy as np


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
    users_k=np.zeros(len(dataset_users))
    for user_id in dataset_users:
        predictions = model.get_user_predictions(user_id)
        # last index is the movie name with the highest rating
        indexes = np.argsort(predictions)
        percentiles = np.zeros(len(indexes))
        # going over movies from last to first (highest rating to lowest)
        for position, movie_id in enumerate(indexes[::-1]):
            # the percentile is the place of that movie divided by the len of all movies
            percentile = 1 - (position / len(indexes))
            # the array percentiles indexes are the movies, the values are the percentile of the movie.
            percentiles[movie_id] = percentile

        ground_truth = create_ground_truth(dataset_users[user_id], k)

        # for each movie count in ground truth (size k) add that movie rank
        sum_percentiles = 0
        for index, item in enumerate(ground_truth):
            sum_percentiles += percentiles[item-1]

        # divide by number of movies in ground truth (k in our case)
        sum_percentiles = sum_percentiles / len(ground_truth)

        users_k[user_id-1] = sum_percentiles

    return np.mean(users_k)


def create_ground_truth(user_data, k):
    sorted_by_rate = sorted(user_data, key=lambda tup: tup[1])
    # keeping above 4 or 3.5 sometimes gives 0 movies...
    sorted_above_4 = [item for item in sorted_by_rate if item[1] >= 4]
    if len(sorted_above_4) == 0:
    # instead we will use top 25 movies in user's true ratings
        sorted_above_4 = sorted_by_rate[-25::]

    # sort by movie id
    sorted_above_4 = sorted(sorted_above_4, key=lambda tup: tup[0])
    only_movies = [movie[0] for movie in sorted_above_4[-len(sorted_above_4)::]]

    return only_movies


def create_predicted(dataset, model, user_id, k):
    predictions = model.get_user_predictions(user_id)

    user_rated_movies = np.zeros(len(dataset[user_id]), dtype=np.int)
    for index, movie_rate in enumerate(dataset[user_id]):
        user_rated_movies[index] = int(movie_rate[0] -1)


    only_user_predictions = predictions[user_rated_movies]

    predictions_sorted = np.argsort(only_user_predictions)
    best_k_predicted = predictions_sorted[-k:]
    best_k_predicted_indices = user_rated_movies[best_k_predicted] + 1

    return set(best_k_predicted_indices)
#
def precision_k(dataset_users, dataset_movies, model, k):
    """
    TP = # ground truth intersect predict (top k)
    FP = # results in top k but not in ground truth
    p@k = TP / (TP+FP) = TP / k
    """
    users_prec_k=np.zeros(len(dataset_users))
    users_reca_k=np.zeros(len(dataset_users))

    for user_id in dataset_users:
        # print (user_id)
        ground_truth_set = create_ground_truth(dataset_users[user_id], k)
        ground_truth_set = set(ground_truth_set)

        # maybe iterate over all movies? keep data of all movies??
        predicted_set = create_predicted(dataset_users, model, user_id, k)
        # because set.intersection would return ground truth after removing predicted from it
        tp = ground_truth_set.intersection(predicted_set)
        fp = predicted_set.difference(ground_truth_set)
        tn = ground_truth_set.difference(predicted_set)

        # tp + fp = k
        prec_k = len(tp) / k
        # tp + tn = ground truth size = 25 in our case (top 25 movies)
        reca_k = len(tp) / (len(tp) + len(tn))


        users_prec_k[user_id-1] = prec_k
        users_reca_k[user_id-1] = reca_k

    return users_prec_k, np.mean(users_prec_k), users_reca_k, np.mean(users_reca_k)


def recall_k(dataset_users, dataset_movies, model, k):
    """
    TP = # ground truth intersect predict (top k)
    TN = # results in ground truth but not in top k
    N = total # of results in ground truth
    R@k = TP / (TP+TN) = TP / N
    """
    users_k=np.zeros(len(dataset_users))
    for user_id in dataset_users:
        ground_truth_set = create_ground_truth(dataset_users[user_id], k)
        ground_truth_set = set(ground_truth_set)

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

    return users_k ,np.mean(users_k)


def map(users_prec_k, users_recall_k, dataset_users):
    results_array = np.zeros(len(dataset_users))
    users_num_of_rating_array=np.zeros(len(dataset_users))
    for user in dataset_users:
        users_num_of_rating_array[user-1] = len(dataset_users[user])

    results_array = (users_prec_k * users_recall_k) / users_num_of_rating_array

    return np.mean(results_array)


def create_ranked_items_for_users(model, dataset_users, h):
    """
    for each user output h ranked items
        pass
    """
    users_h_ratings={}
    for user_id in dataset_users:
        predictions = model.get_user_predictions(user_id)
        # last index is the movie name with the highest rating
        indexes = np.argsort(predictions)

        h_items = np.zeros(len(indexes))
        # going over movies from last to first (highest rating to lowest)
        for rank, movie_id in enumerate(indexes[-h::]):
            h_items.append((rank, movie_id))

        users_h_ratings[user_id] = h_items

    return users_h_ratings


def create_metrices_for_user(users_h_ratings, k):
    """
    output metrices per user
    """
    return


# def human_readable_output(dataset_users, model, user_id, h):
    """
    output user history votes with names of items on training set
    output top h items recommended
    """
    # for movie_rating in dataset_users[user_id]:
        # get name of movie by id
        # print (movie_rating)
    #
    # predictions = model.get_user_predictions(user_id)
    # # last index is the movie name with the highest rating
    # indexes = np.argsort(predictions)
    # top_h_movie = indexes[-h::]
    #
    # # sort by movie id
    # top_h_movie = sorted(top_h_movie)
    # print (top_h_movie)
    # return


def run_metrices(dataset, model, k, size_of_data):
    print ("--- Running metrices for " + dataset["name"] + " ---")
    mpr = mean_percentile_rank(dataset["users"], model, k)
    se_score = se(dataset["users"], model)
    rmse_score = rmse(dataset["users"], model, size_of_data)
    users_prec_k, prec_k, users_recall_k, reca_k = precision_k(dataset["users"],dataset["movies"] , model, k)
    mean_avg_precision = map(users_prec_k, users_recall_k, dataset["users"])

    print ("MPR:{}, SE:{}, RMSE:{}\nP@k:{}, R@k:{}, MAP:{}".format(mpr, se_score, rmse_score, prec_k, reca_k, mean_avg_precision))

    print ("#"*80 + "\n")

if __name__ == "__main__":
    pass