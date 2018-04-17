from collections import namedtuple
import numpy as np
MovieData = namedtuple('MovieData', 'name genres')

def generate_movies_data_dict(file_path):
    movies_dict = {}
    with open(file_path, mode='rb') as movies_data:
        movie_lines = movies_data.readlines()
        for line in movie_lines:
            id, name, geners = parse_movie_line(line[:-1])
            movies_dict[id] = MovieData(name, geners)

    return movies_dict

def parse_movie_line(line):
    line = line.decode("utf-8", 'replace')

    id, name, genres = line.split('::')
    genres = genres.split('|')

    return id, name, genres


def generate_users_dict(file_path, num_ratings=-1):
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
            users_dict[int(user_id)].append((int(movie_id), int(rating)))

    return users_dict

# def generate_movies_dict(file_path):
#     movies_dict = {}
#     with open(file_path, mode='rb') as user_ratings:
#         # Remove the description line.
#         user_ratings.readline()
#
#         rating_lines = user_ratings.readlines()
#         for line in rating_lines:
#             user_id, movie_id, rating = parse_rating_line(line)
#             if int(movie_id) not in movies_dict:
#                 movie_id = int(movie_id)
#                 movies_dict[movie_id] = []
#             movies_dict[int(movie_id)].append((int(user_id), int(rating)))
#
#     return movies_dict

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

if __name__ == "__main__":
    movies_data = generate_movies_data_dict("../data/movies.dat")
    users = generate_users_dict("../data/ratings.dat")
    movies = generate_movies_dict_from_users_dict(users)
    # movies = generate_movies_dict("../data/ratings.dat")
    matrix = generate_ratings_matrix(users ,movies)

    print(matrix[:25][:25])

