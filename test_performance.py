import h5py
from neural_net_module import neuralnet
import numpy as np
import matplotlib.pyplot as plt


# displays an array of images
def show_image(images, index_list, classes):
    for index in index_list:
        image = images[:, [index]]
        images_class = classes[:, [index]]

        image = image.reshape((64, 64, 3))

        # printing class
        print("class: " + str(images_class))
        print()

        plt.imshow(image)
        plt.show()


# rounds sigmoid output
def binary(x):
    return np.where(x > 0.5, 1, 0)


# converts softmax out put to one hot format
def one_hot(x):
    max_per_col = np.max(x, axis=0, keepdims=True)
    return np.where(x == max_per_col, 1, 0)


def get_prediction(model, X, Y):
    neural_net = neuralnet.NeuralNetwork(model)
    neural_net.load_input(X)

    predictions = neural_net.fore_prop(neural_net.X)

    # rounding predictions
    predictions = binary(predictions)

    return predictions


def get_accuracy(model, X, Y):
    predictions = get_prediction(model, X, Y)

    # calculating accuracy
    matches = np.all(predictions == Y, axis=0)

    accuracy = np.mean(matches)

    return accuracy


def plot_cost(training_cache_file):
    # cost log
    with np.load(training_cache_file, "r", allow_pickle=True) as data:
        cost_log = data["cost_log"].tolist()

    # plotting cost
    plt.plot(cost_log)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.show()


# loading datasets
with h5py.File("datasets/datasets.h5", "r") as datasets:
    training_set_X = datasets["training_set_X"][:]
    training_set_Y = datasets["training_set_Y"][:]

    dev_set_X = datasets["dev_set_X"][:]
    dev_set_Y = datasets["dev_set_Y"][:]

    test_set_X = datasets["test_set_X"][:]
    test_set_Y = datasets["test_set_Y"][:]

# loading neural network
model_file = r"parameters/final_model_params.npz"
training_cache_file = r"parameters/final_model_training_cache.npz"

# computing accuracy
training_set_accuracy = get_accuracy(model_file, training_set_X, training_set_Y)
test_set_accuracy = get_accuracy(model_file, test_set_X, test_set_Y)
dev_set_accuracy = get_accuracy(model_file, dev_set_X, dev_set_Y)

# printing result
print("test set accuracy: {0}%".format(test_set_accuracy * 100))
print("dev set accuracy: {0}%".format(dev_set_accuracy * 100))
print("training set accuracy: {0}%".format(training_set_accuracy * 100))

plot_cost(training_cache_file)

# # showing miss classified images
# predictions = get_prediction(model_file, test_set_X, test_set_Y)
#
# miss_classified = np.where(predictions != test_set_Y)[1]

# show_image(test_set_X, miss_classified, predictions)
