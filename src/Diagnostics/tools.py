import matplotlib.pyplot as plt
from Diagnostics.pdf_generator import PDFGenerator

DELIVERABLE_2_FILE = "../deliverable/deliverable_2_data"
DELIVERABLE_3_FILE = "../deliverable/deliverable_3_data"
DELIVERABLE_4_FILE = "../deliverable/deliverable_4_data"


# =========================================== Helpers ==================================================================

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

def parse_plot_file(file_name):
    x_values = []
    y_values = []
    with open(file_name) as file:
        content = file.readlines()
        for line in content:
            x, y = line.split("::")
            x_values.append(x)
            y_values.append(y)

    return x_values, y_values

def parse_double_plot_file(file_name,):
    x_values = []
    y_values = []
    z_values = []
    with open(file_name) as file:
        content = file.readlines()
        for line in content:
            x, y, z = line.split("::")
            x_values.append(x)
            y_values.append(y)
            z_values.append(z)

    return x_values, y_values, z_values


# =========================================== Deliverables generation ==================================================
def generate_deliverable_one():
    file_name = "{}_test_train_error_plot".format()
    plt.savefig(file_name)
    plt.clf()

def generate_deliverable_two():
    d_values, m1_values, m2_values = parse_double_plot_file(DELIVERABLE_2_FILE)
    plt.plot(d_values, m1_values)
    plt.ylabel("Rmse")
    plt.xlabel("log lamda value")

    plt.xscale('log')

    file_name = "deliverable_two_metric_one"
    plt.savefig(file_name)
    plt.clf()

    plt.plot(d_values, m2_values)
    plt.ylabel("Recall@10")
    plt.xlabel("log lamda value")

    plt.xscale('log')
    file_name = "deliverable_two_metric_two"
    plt.savefig(file_name)
    plt.clf()

def generate_deliverable_three():
    d_values, m1_values, m2_values = parse_double_plot_file(DELIVERABLE_3_FILE)

    plt.plot(d_values, m1_values)
    plt.ylabel("Rmse")
    plt.xlabel("D(latent variables) value")

    file_name = "deliverable_three_metric_one"
    plt.savefig(file_name)
    plt.clf()

    plt.plot(d_values, m2_values)
    plt.ylabel("Recall@10")
    plt.xlabel("D(latent variables) value")

    file_name = "deliverable_three_metric_two"
    plt.savefig(file_name)
    plt.clf()

def generate_deliverable_four():
    x_values, y_values = parse_plot_file(DELIVERABLE_4_FILE)

    plt.plot(x_values, y_values)
    plt.title("Running Time")

    plt.xscale('log')

    plt.ylabel("Running Time")
    plt.xlabel("D(latent variables) value")

    file_name = "deliverable_four"
    plt.savefig(file_name)
    plt.clf()


if __name__ == "__main__":
    # print("Error file parsed = {}".format(parse_error_file("mock_data")))
    # sgd_als_errors_plots(["mock_data", "mock_data_2"], ["mock_data", "mock_data_2"], "hyperparameters")
    # generate_deliverable_four(DELIVERABLE_4_FILE)
    generate_deliverable_three()

