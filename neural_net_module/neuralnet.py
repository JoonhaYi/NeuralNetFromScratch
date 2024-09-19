import numpy as np
from neural_net_module import algorithms


class NeuralNetwork:
    def __init__(self, model_file):
        # declaring constants
        self.model_file = model_file

        # default hyperparameters
        self.hyperparams = {
            "hidden_layer_activ_func": algorithms.Functions.relu,
            "batch_norm": False
        }

        # input matrix
        self.X = None

        # number of inputs
        self.m = None

        # number of layers
        self.L = None

        # declaring variables
        self.params = None
        self.output = []

        # cache for back prop
        self.fore_prop_cache = []

        # batch norm cache
        self.activation_stats = []

        # loading saved model (returns true if successfully loaded)
        self.is_loaded = self.__load_model()

    # initialization
    def __load_model(self):
        try:
            # loading parameters
            # nn hyperparam = non training hyperparameters that are required for the model to run
            with np.load(self.model_file, "r", allow_pickle=True) as data:
                self.params = data["params"].tolist()
                self.hyperparams = data["non_training_hyperparams"].tolist()
                self.activation_stats = data["stats"]

                self.L = len(self.hyperparams["layer_dims"]) - 1

                return True

        except FileNotFoundError:
            return False

    # dataset configuration
    @staticmethod
    def compute_stats(x):
        # computing mean and std along the rows
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)

        return {"mean": mean, "std": std}

    @staticmethod
    def normalize(x, stats):
        # subtracting mean
        x = x - stats["mean"]

        # dividing by std
        x = x / stats["std"]

        return x

    def load_input(self, X, do_normalize=True):
        self.X = X
        self.m = X.shape[1]

        # normalizing the dataset
        if do_normalize:
            self.X = self.normalize(X, self.activation_stats[0])

    # forward propagation
    def fore_prop(self, X):
        # resetting cache
        self.fore_prop_cache = []

        # propagating forward
        A = X

        for l in range(1, self.L):
            A = self.pass_forward(l, A, self.hyperparams["hidden_layer_activ_func"])

        # calculating output
        A = self.pass_forward(self.L, A, self.hyperparams["output_layer_activ_func"])
        self.output = A

        return A

    def pass_forward(self, l, A_prev, g):
        # retrieving parameters
        W = self.params["W" + str(l)]
        b = self.params["b" + str(l)]

        # creating cache
        cache = {"A_prev": A_prev}

        # dotting weights
        Z = np.dot(W, A_prev)

        # batch normalization
        if self.hyperparams["batch_norm"] and l != self.L:
            gamma = self.params["gamma" + str(l)]

            # normalizing Z
            Z_norm, mean, std = self.batch_norm(l, Z)

            # caching
            cache["Z_orig"] = Z
            cache["Z_norm"] = Z_norm
            cache["mean"] = mean
            cache["std"] = std

            # multiplying gamma
            Z = Z_norm * gamma

        # adding bias
        Z = Z + b

        # caching Z
        cache["Z"] = Z

        # activation function
        A = g({"input": Z})

        # appending cache
        self.fore_prop_cache.append(cache)

        return A

    def batch_norm(self, l, Z):
        # retrieving cached mean and std
        stats = self.activation_stats[l]

        # normalizing Z
        Z_norm = self.normalize(Z, stats)

        return Z_norm, stats["mean"], stats["std"]
