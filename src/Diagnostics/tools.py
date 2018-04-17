import matplotlib.pyplot as plt
from Diagnostics.pdf_generator import PDFGenerator

def plot_train_and_test_vs_iteration(name, train_data, test_data, save=True):
    # Plot the test and the train data errors.
    plt.plot(test_data)
    plt.plot(train_data)

    # Add plot generic information.
    plt.title(name)
    plt.ylabel("Mean Square Error")
    plt.xlabel("Number of iteration")
    plt.legend(["Train Data", "Test Data"])

    if save:
        file_name = "{}_test_train_error_plot".format(name)
        plt.savefig(file_name)
        plt.clf()
        return file_name + ".png"
    else :
        plt.show()
        plt.clf()


def parse_error_file(error_file):
    errors = []
    with open(error_file) as file:
        content = file.readlines()
        for line in content:
            errors.append(float(line))

    return errors


def sgd_als_errors_plots(sgd_files, als_files, hyperparameters, save=True):
    sgd_train = parse_error_file(sgd_files[0])
    sgd_test = parse_error_file(sgd_files[1])
    sgd_plot = plot_train_and_test_vs_iteration("SGD_ERROR", sgd_train, sgd_test, save)

    als_train = parse_error_file(als_files[0])
    als_test = parse_error_file(als_files[1])
    als_plot = plot_train_and_test_vs_iteration("ALS_ERROR", als_train, als_test, save)

    if save:
        pdf = PDFGenerator("Delivrable_1.pdf")
        pdf.write_text("Hypereparamters : lambda=1 k=2")
        pdf.add_image(sgd_plot)
        pdf.add_image(als_plot)
        pdf.close()



def test_plot():
    train_error_mock = [2, 1.3, 1.1, 0.8, 0.67, 0.51, 0.44, 0.27, 0.11, 0.02]
    test_error_mock = [2.3, 1.5, 1.4, 0.95, 0.88, 0.67, 0.64, 0.57, 0.51, 0.42]
    plot_train_and_test_vs_iteration("SGD_Error", train_error_mock, test_error_mock, True)

if __name__ == "__main__":
    print("Error file parsed = {}".format(parse_error_file("mock_data")))
    sgd_als_errors_plots(["mock_data", "mock_data_2"], ["mock_data", "mock_data_2"], "hyperparameters")
