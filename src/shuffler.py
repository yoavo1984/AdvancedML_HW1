import random


def shuffle_array(array):
    """
    Returns a shuffled copy of the given array.
    :param array: The array we want a shuffled copy of.
    :return: A shuffled copy of the given array
    """
    array_copy = array.copy()
    random.shuffle(array_copy)
    return array_copy


def split_data_randomly(data_object):
    """
    Divides the data into 80% training set and 20% test set.
    Data is splitted s.t. each user has both data points in the training set and data set 
    :param data_object: 
    :return: Training set and test set. 
    """
    training_set = {}
    test_set = {}
    for user_id, ratings in data_object.items():
        num_ratings = len(ratings)
        num_training = int(0.8 * num_ratings)

        shuffled_ratings = shuffle_array(ratings)
        training_ratings = shuffled_ratings[0:num_training]
        test_ratings = shuffled_ratings[num_training:]

        training_set[user_id] = training_ratings
        test_set[user_id] = test_ratings

    return training_set, test_set

if __name__ == "__main__":
    mock_data = {'1' : [(1,1),(2,2),(3,3),(4,4),(5,5)],
                 '2' : [(1,1),(2,2),(3,3),(4,4),(5,5)],
                 '3' : [(1,1),(2,2),(3,3),(4,4),(5,5)]}
    print (split_data_randomly(mock_data))