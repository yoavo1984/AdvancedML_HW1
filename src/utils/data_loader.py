from collections import namedtuple
import numpy as np
MovieData = namedtuple('MovieData', 'name genres')

# Doing some changes in the utils folder.

def generate_movies_data_dict(file_path):
    movies_dict = {}
    with open(file_path, mode='rb') as movies_data:
        movie_lines = movies_data.readlines()
        for line in movie_lines:
            id, name, geners = parse_movie_line(line[:-1])
            movies_dict[int(id)] = MovieData(name, geners)

    return movies_dict

def parse_movie_line(line):
    line = line.decode("utf-8", 'replace')

    id, name, genres = line.split('::')
    genres = genres.split('|')

    return id, name, genres


def generate_users_dict(file_path, missing_movie_id_index, num_ratings=-1):
    users_dict = {}
    with open(file_path, mode='rb') as user_ratings:
        # Remove the description line.
        user_ratings.readline()

        rating_lines = user_ratings.readlines()
        if num_ratings != -1:
            rating_lines = rating_lines[:num_ratings]
        for line in rating_lines[0:]:
            user_id, movie_id, rating = parse_rating_line(line)
            if int(user_id) not in users_dict:
                user_id = int(user_id)
                users_dict[user_id] = []
            if int(movie_id) in missing_movie_id_index:
                movie_id = missing_movie_id_index[int(movie_id)]
            users_dict[int(user_id)].append((int(movie_id), int(rating)))

    return users_dict


def generate_movies_dict_from_users_dict(users_dict):
    movies_dict = {}
    for user_id in users_dict:
        for movie_rate in users_dict[user_id]:
            movie, rate = *movie_rate,
            if movie not in movies_dict:
                movies_dict[movie] = []

            movies_dict[movie].append((user_id, rate))
    return movies_dict


def parse_rating_line(line):
    line = line.decode("utf-8", 'replace')
    user_id, movie_id, rating, _ = line.split("::")
    return user_id, movie_id, rating


def generate_ratings_matrix(users_dict, movies_dict):
    matrix = np.zeros((len(users_dict), 3953))

    for user_id, user_ratings in users_dict.items():
        user_id = int(user_id)
        for rating in user_ratings:
            movie_id = rating[0]
            movie_rating = rating[1]
            matrix[user_id - 1][movie_id - 1] = movie_rating

    return matrix

def calculate_dataset_mu(dataset):
    # maybe use calculate_size_of_data_set
    sum_of_rates = 0
    num_of_rates = 0
    for key in dataset:
        for movie_rate in dataset[key]:
            rate = movie_rate[1]
            sum_of_rates += rate
            num_of_rates += 1
    return round(float(sum_of_rates / num_of_rates), 2)


def calculate_size_of_data_set(dataset):
    num_of_rates = 0
    for key in dataset:
        for movie_rate in dataset[key]:
            num_of_rates += 1
    return num_of_rates


def generate_movie_missing_ids_indices(file_path):
    with open(file_path, mode='rb') as user_ratings:
        rating_lines = user_ratings.readlines()
        missing_ids = []
        offset = 0
        last_id = 0
        for index, line in enumerate(rating_lines[0:]):
            id, _, _ = parse_movie_line(line)
            if int(id)-1 != index+offset:
                offset += 1
                missing_ids.append(index+offset)
            last_id = id

    index_dict = {}
    for index, value in enumerate(missing_ids):
        index_dict[int(last_id)-index] = value

    return index_dict


if __name__ == "__main__":
    # missing_movie_index_dict = find_missing_id("../data/movies.dat")

    movies_data = generate_movies_data_dict("../data/movies.dat")
    users = generate_users_dict("../data/ratings.dat", missing_movie_index_dict)

    movies = generate_movies_dict_from_users_dict(users)
    # movies = generate_movies_dict("../data/ratings.dat")
    matrix = generate_ratings_matrix(users ,movies)

