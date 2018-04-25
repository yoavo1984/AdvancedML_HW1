import math

import numpy as np

from utils import data_loader


def se(dataset, model):
    sum = 0
    for user_id in dataset:
        for movie_rate in dataset[user_id]:
            movie_id = movie_rate[0]
            movie_true_rating = movie_rate[1]
            prediction = model.predict(user_id, movie_id)
            sum += pow(movie_true_rating - prediction, 2)

    return sum


def rmse(dataset, model, size_of_data=0):
    # calculate squared error
    sum = 0
    size_of_data = 0
    # need to compute one time and reuse

    for user_id in dataset:
        for movie_rate in dataset[user_id]:
            movie_id = movie_rate[0]
            movie_true_rating = movie_rate[1]
            prediction = model.predict(user_id, movie_id)
            sum += pow(movie_true_rating - prediction, 2)
            size_of_data += 1

    # take the mean
    sum = sum / size_of_data

    # take the root
    sum = math.sqrt(sum)

    return sum


def mean_percentile_rank(dataset_users, model):
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

        ground_truth = create_ground_truth(dataset_users[user_id])

        # for each movie count in ground truth (size k) add that movie rank
        sum_percentiles = 0
        for index, item in enumerate(ground_truth):
            sum_percentiles += percentiles[item-1]

        # divide by number of movies in ground truth (k in our case)
        sum_percentiles = sum_percentiles / len(ground_truth)

        users_k[user_id-1] = sum_percentiles

    return np.mean(users_k)


def create_ground_truth(user_data):
    sorted_by_rate = sorted(user_data, key=lambda tup: tup[1])
    # keeping above 4 or 3.5 sometimes gives 0 movies...
    sorted_above_4 = [item for item in sorted_by_rate if item[1] >= 4]
    if len(sorted_above_4) == 0:
    # instead we will use top 25 movies in user's true
        sorted_above_4 = sorted_by_rate[-25::]

    # trying to take only k movies for ground truth
    # sorted_above_4 = sorted_by_rate[-k::]

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


def recall_k(dataset_users, model, k):
    _, _, users_recall_k, reca_k = calc_precision_recall_k(dataset_users, model, k)
    return users_recall_k, reca_k


def precision_k(dataset_users, model, k):
    users_prec_k, prec_k, _, _ = calc_precision_recall_k(dataset_users, model, k)
    return users_prec_k, prec_k


def calc_precision_recall_k(dataset_users, model, k):
    """
    TP = # ground truth intersect predict (top k)
    FP = # results in top k but not in ground truth
    TN = # results in ground truth but not in top k
    p@k = TP / (TP+FP) = TP / k
    r@k = TP / (TP+TN) = TP / N
    """
    users_prec_k=np.zeros(len(dataset_users))
    users_reca_k=np.zeros(len(dataset_users))

    for user_id in dataset_users:
        # print (user_id)
        ground_truth_set = create_ground_truth(dataset_users[user_id])
        ground_truth_set = set(ground_truth_set)

        # maybe iterate over all movies? keep data of all movies??
        predicted_set = create_predicted(dataset_users, model, user_id, k)
        # because set.intersection would return ground truth after removing predicted from it
        tp = ground_truth_set.intersection(predicted_set)
        fn = predicted_set.difference(ground_truth_set)
        tn = ground_truth_set.difference(predicted_set)

        # tp + fp = k
        prec_k = len(tp) / k
        # tp + tn = ground truth size
        reca_k = len(tp) / len(ground_truth_set)


        users_prec_k[user_id-1] = prec_k
        users_reca_k[user_id-1] = reca_k

    return users_prec_k, np.mean(users_prec_k), users_reca_k, np.mean(users_reca_k)

def single_user_precision_recall(dataset_users, user_id, model, k):
    """
    TP = # ground truth intersect predict (top k)
    FP = # results in top k but not in ground truth
    TN = # results in ground truth but not in top k
    p@k = TP / (TP+FP) = TP / k
    r@k = TP / (TP+TN) = TP / N
    """

    # print (user_id)
    ground_truth_set = create_ground_truth(dataset_users[user_id])
    ground_truth_set = set(ground_truth_set)

    predicted_set = create_predicted(dataset_users, model, user_id, k)

    tp = ground_truth_set.intersection(predicted_set)
    fn = predicted_set.difference(ground_truth_set)
    tn = ground_truth_set.difference(predicted_set)

    prec_k = len(tp) / k
    reca_k = len(tp) / len(ground_truth_set)

    return prec_k, reca_k


def map(dataset_users, model):
    # sum1 is the lecture formula
    # sum2 is the internet formula
    def average_preciison(n, num_of_user_ratings):
        # n+1 because we include n value
        sum1 = 0
        sum2 = 0
        prev_reca_k = 0
        for i in range(1, n+1):
            prec_k, reca_k = single_user_precision_recall(dataset_users, user_id, model, i)

            change_in_recall = reca_k-prev_reca_k
            sum1 += prec_k*change_in_recall
            prev_reca_k = reca_k

            sum2 += prec_k

        # divide the sum buy the sum of relevant items a.k.a # of movies above 4..
        # in our case we will choose this number to be n
        # so if we are right at all iterations of p@k up to n we will get score of 1
        # sum1 does not need to be divided by the number of reco. because the recall diff is making this the same
        sum1 = sum1
        sum2 = sum2 / n
        return sum1, sum2

    sum1 = 0
    sum2 = 0
    for user_id in dataset_users:
        ground_truth_set = create_ground_truth(dataset_users[user_id])
        AP1, AP2 = average_preciison(len(ground_truth_set), len(dataset_users[user_id]))
        sum1 += AP1
        sum2 += AP2


    sum1 = sum1 / len(dataset_users)
    sum2 = sum2 / len(dataset_users)


    return sum1, sum2


def print_ranked_items(model, dataset_users, h):
    """
    for each user output h ranked items
    """
    users_h_ratings={}
    for user_id in dataset_users:
        predictions = model.get_user_predictions(user_id)
        # last index is the movie name with the highest rating
        indexes = np.argsort(predictions)

        h_items = []

        # going over movies from last to first (highest rating to lowest)
        for rank, movie_id in enumerate(indexes[-h::]):
            h_items.append(movie_id)
            h_items = h_items[::-1]

        users_h_ratings[user_id] = h_items

    return users_h_ratings


def create_metrices_for_user(users_h_ratings, k):
    """
    output metrices per user
    """
    return


def human_readable_output(dataset, model, user_id, h, history_flag):
    """
    output user history votes with names of items on training set
    output top h items recommended
    """
    dataset_users = dataset.get_train_dataset()['users']
    movies_dict = data_loader.generate_movies_data_dict("../data/movies.dat")
    missing_movies_id = dataset.missing_movies_ids_dict
    oppo_missing_movies_id = new_dict = dict (zip(missing_movies_id.values(),missing_movies_id.keys()))



    if history_flag == 1:
        print("Printing history rates for user {0}".format(user_id))
        print("#" * 80)
        for movie_rating in dataset_users[user_id]:
            movie_id = movie_rating[0]
            movie_rate = movie_rating[1]
            if movie_id in oppo_missing_movies_id:
                movie_id = oppo_missing_movies_id[movie_id]

            movie_name = movies_dict[movie_id][0]
            movie_rate = movie_rate

            print ("{0}:{1}".format(movie_name, movie_rate))

    predictions = model.get_user_predictions(user_id)
    # last index is the movie name with the highest rating
    indexes = np.argsort(predictions)
    top_h_movies = indexes[-h:]

    # sort by movie id
    top_h_movies = sorted(top_h_movies)
    print ("Printing top {0} movies for user {1}".format(h, user_id))
    print ("#"*80)
    for movie_id in top_h_movies[::-1]:
        if movie_id+1 in oppo_missing_movies_id:
            movie_id = oppo_missing_movies_id[movie_id+1]
            print ("{0}".format(movies_dict[movie_id][0]))

        else:
            print ("{0}".format(movies_dict[movie_id+1][0]))
    print ("#"*80)

    return


def run_metrices(dataset, model, k, size_of_data, output_ranked_items):
    print ("--- Running metrices for " + dataset["name"] + " with k = {}".format(k) + " ---")
    mpr = mean_percentile_rank(dataset["users"], model, k)
    se_score = se(dataset["users"], model)
    rmse_score = rmse(dataset["users"], model, size_of_data)
    users_prec_k, prec_k, users_recall_k, reca_k = calc_precision_recall_k(dataset["users"] , model, k)
    mean_avg_precision_lecture, mean_avg_precision_internet = map(dataset["users"], model)
    print ("MPR:{}, SE:{}, RMSE:{}\nP@k:{}, R@k:{}\nMAP_Lecture:{}, MAP_Internet:{}\n".format(mpr,
                                                                                              se_score,
                                                                                              rmse_score,
                                                                                              prec_k,
                                                                                              reca_k,
                                                                                              mean_avg_precision_lecture,
                                                                                              mean_avg_precision_internet))

    if output_ranked_items == 1:
        print_ranked_items(model, dataset["users"], 2)

def get_part_six_metrices_output(dataset, model):
    model.generate_prediction_matrix()
    rmse_s                   = rmse(dataset["users"], model)
    mpr                      = mean_percentile_rank(dataset["users"], model)
    _, prec_two, _, reca_two = calc_precision_recall_k(dataset["users"], model, 2)
    _, prec_ten, _, reca_ten = calc_precision_recall_k(dataset["users"], model, 10)
    mean_avg_precision, _    = map(dataset["users"], model)

    return "Evaluation Metrics:\n" \
           "-------------------\n" \
           "- RMSE = {}\n" \
           "- MPR  = {}\n" \
           "- P@2  = {}\n" \
           "- P@10 = {}\n" \
           "- R@2  = {}\n" \
           "- R@10 = {}\n" \
           "- MAP  = {}\n\n" \
           "".format(round(rmse_s,4),
                     round(mpr,4),
                     round(prec_two,4),
                     round(prec_two,4),
                     round(prec_ten,4),
                     round(reca_two,4),
                     round(reca_ten,4))



if __name__ == "__main__":
    pass