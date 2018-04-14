from collections import namedtuple
import numpy as np
MovieData = namedtuple('MovieData', 'name genres')

def generate_movies_dict(file_path):
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


def generate_users_dict(file_path):
    users_dict = {}
    with open(file_path, mode='rb') as user_ratings:
        # Remove the description line.
        user_ratings.readline()

        rating_lines = user_ratings.readlines()
        for line in rating_lines:
            user_id, movie_id, rating = parse_rating_line(line)
            if user_id not in users_dict:
                users_dict[user_id] = []
            users_dict[user_id].append((int(movie_id), int(rating)))

    return users_dict

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


if __name__ == "__main__":
    movies = generate_movies_dict("../data/movies.dat")
    users = generate_users_dict("../data/ratings.dat")
    matrix = generate_ratings_matrix(users ,movies)

    print(matrix[:25][:25])

