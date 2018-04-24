from utils import data_loader, shuffler


class Dataset():
    def __init__(self, movie_file_path, users_file_path, number_of_items=-1):
        self.movie_file_path = movie_file_path
        self.users_file_path = users_file_path

        self.missing_movies_ids_dict = data_loader.generate_movie_missing_ids_indices(self.movie_file_path)

        self.users_dict = data_loader.generate_users_dict(self.users_file_path, self.missing_movies_ids_dict, number_of_items)
        self.movies_dict = data_loader.generate_movies_data_dict(self.movie_file_path)

        self.train_data, self.test_data = shuffler.split_data_randomly(self.users_dict)

    def get_train_dataset(self):
        ret_dataset = {}
        ret_dataset['users'] = self.train_data
        ret_dataset['movies'] = data_loader.generate_movies_dict_from_users_dict(self.train_data)
        ret_dataset['name'] = "Train Dataset"
        return ret_dataset

    def get_test_dataset(self):
        ret_dataset = {}
        ret_dataset['users'] = self.test_data
        ret_dataset['movies'] = data_loader.generate_movies_dict_from_users_dict(self.test_data)
        ret_dataset['name'] = "Test Dataset"
        return ret_dataset

    def get_number_of_users(self):
        return len(self.users_dict)

    def get_number_of_movies(self):
        return len(self.movies_dict)

    def get_test_dataset_size(self):
        return self.calculate_size_of_data_set(self.test_data)

    def get_train_dataset_size(self):
        return self.calculate_size_of_data_set(self.train_data)

    @staticmethod
    def get_dataset_mean_rating(dataset):
        users_dataset = dataset['users']
        sum_of_rates = 0
        num_of_rates = 0
        for key in users_dataset:
            for movie_rate in users_dataset[key]:
                rate = movie_rate[1]
                sum_of_rates += rate
                num_of_rates += 1
        return round(float(sum_of_rates / num_of_rates), 2)

    @staticmethod
    def calculate_size_of_data_set(dataset):
        num_of_rates = 0
        for key in dataset:
            num_of_rates += len(dataset[key])
        return num_of_rates