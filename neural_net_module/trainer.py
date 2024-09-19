from neural_net_module import neuralnet, algorithms
import numpy as np

nn = neuralnet.NeuralNetwork


class Trainer(nn):
    def __init__(self, model_file, training_cache):
        # initializing Neuralnetwork class
        neuralnet.NeuralNetwork.__init__(self, model_file)

        # declaring constants
        self.training_cache = training_cache

        # default initialization methods
        self.default_initializers = {
            "hidden_layer_initializer": algorithms.Functions.he,
            "output_layer_initializer": algorithms.Functions.xavier
        }

        # default hyperparameters
        self.hyperparams.update({
            "lambd": 0,
            "keep_prob": 1,
            "optimizer": algorithms.Functions.adam,
            "beta1": 0.9,
            "beta2": 0.999,
            "decay_function": lambda dic: dic["learning_rate"],
            "decay_rate": 0,
            "decay_interval": -1,
        })

        # labeling matrix
        self.Y = None

        # declaring variables
        self.grads = {}
        self.cost_log = []
        self.t = 0

        # exponentially weighted averages (momentum and RMS)
        self.v = None
        self.s = None

        # loading training cache
        self.is_cache_loaded = self.__load_training_cache()

    # overridden methods
    def pass_forward(self, l, A_prev, g):
        A = nn.pass_forward(self, l, A_prev, g)

        # adding drop out
        if self.hyperparams["keep_prob"] < 1 and l != self.L:
            # generating random matrix D with the same dimensions as A
            D = np.random.rand(A.shape[0], A.shape[1])

            # rounding D to matrix of 0s and 1s according to keep_prop
            D = (D < self.hyperparams["keep_prob"]).astype(int)

            # multiplying A and D element wise
            A = A * D

            # dividing A by keep_prop to bring A back to scale
            A = A / self.hyperparams["keep_prob"]

            # caching D
            self.fore_prop_cache[l - 1]["D"] = D

        return A

    def batch_norm(self, l, Z, B=0.9):
        # calculating mean and standard deviation
        stats = self.compute_stats(Z)

        # subtracting mean and dividing by std
        Z_norm = self.normalize(Z, stats)

        # computing and caching exponentially weighted averages of mean and std
        self.activation_stats[l]["mean"] = B * self.activation_stats[l]["mean"] + (1 - B) * stats["mean"]
        self.activation_stats[l]["std"] = B * self.activation_stats[l]["std"] + (1 - B) * stats["std"]

        return Z_norm, stats["mean"], stats["std"]

    # initialization
    def initialize_model(self, hyperparameters, initializers=None):
        # setting hyperparameters
        self.__set_hyperparams(hyperparameters)

        # initializing new parameters if no model is loaded
        if not self.is_loaded:
            # loading default initialization methods if not specified
            if initializers is None:
                initializers = self.default_initializers

            # initializing parameters
            self.__initialize_params(initializers)

            # initializing batch norm
            self.__initialize_batch_norm()

            # initializing momentum and RMS
            self.v = self.__initialize_adam()
            self.s = self.v.copy()

            # resetting cost log
            self.cost_log = []

    def __set_hyperparams(self, dic):
        # overriding default hyperparameters with custom hyperparameters
        self.hyperparams.update(dic)

        # storing the number of layers into a variable
        self.L = len(self.hyperparams["layer_dims"]) - 1

    def __initialize_params(self, initializers):
        # resetting parameters
        self.params = {}

        layer_dims = self.hyperparams["layer_dims"]

        # initializing hidden layer parameters
        for l in range(1, self.L):
            # initializing weights with the specified hidden layer initialization method
            self.params["W" + str(l)] = initializers["hidden_layer_initializer"](
                {"n": layer_dims[l],
                 "n_prev": layer_dims[l - 1]}
            )

            # initializing bias to zeros
            self.params["b" + str(l)] = np.zeros((layer_dims[l], 1))

            # checking if batch norm is enable and initializing batch norm parameter gamma
            if self.hyperparams["batch_norm"]:
                self.params["gamma" + str(l)] = np.zeros((layer_dims[l], 1)) + 1

        # initializing output layer parameters
        self.params["W" + str(self.L)] = initializers["output_layer_initializer"](
            {"n": layer_dims[self.L],
             "n_prev": layer_dims[self.L - 1]}
        )

        self.params["b" + str(self.L)] = np.zeros((layer_dims[self.L], 1))

    def __initialize_batch_norm(self):
        # looping over each layer
        for i in range(self.L):
            # creating a dictionary for storing mean and std of each activation
            self.activation_stats.append({"mean": 0, "std": 0})

    def __initialize_adam(self):
        dic = {}

        for l in range(1, self.L + 1):
            # initializing momentum and RMS to zero
            dic["dW" + str(l)] = np.zeros_like(self.params["W" + str(l)])
            dic["db" + str(l)] = np.zeros_like(self.params["b" + str(l)])

            # checking for batch norm and initializing batch norm parameter gamma
            if self.hyperparams["batch_norm"] and l != self.L:
                dic["dgamma" + str(l)] = np.zeros_like(self.params["gamma" + str(l)])

        return dic

    # dataset
    def load_dataset(self, X, Y, do_normalize=True):
        # loading labeling matrix Y
        self.Y = Y

        # normalizing X if specified (on by default)
        if do_normalize:
            self.activation_stats[0] = self.compute_stats(X)

        # loading dataset
        nn.load_input(self, X, do_normalize)

        # setting the batch size to m if batch size is not specified
        if "batch_size" not in self.hyperparams.keys():
            self.hyperparams["batch_size"] = X.shape[1]

    # caching
    def save_model(self):
        # saving parameters, momentum, RMS, activation stats and hyperparameters required for fore prop
        np.savez(
            self.model_file,
            params=self.params,
            non_training_hyperparams=self.__get_non_training_hyperparams(),
            stats=self.activation_stats,
        )

    def __get_non_training_hyperparams(self):
        # retrieving non training hyperparameters needed for forward propagation
        non_training_hyperparam = {
            "layer_dims": self.hyperparams["layer_dims"],
            "hidden_layer_activ_func": self.hyperparams["hidden_layer_activ_func"],
            "output_layer_activ_func": self.hyperparams["output_layer_activ_func"],
            "batch_norm": self.hyperparams["batch_norm"]
        }

        return non_training_hyperparam

    def save_training_cache(self):
        # caching t, momentum, RMS and cost
        np.savez(
            self.training_cache,
            time_stamp=self.t,
            EMAs=(self.v, self.s),
            cost_log=self.cost_log
        )

    def __load_training_cache(self):
        try:
            # loading saved values of t, momentum, RMS and cost
            with np.load(self.training_cache, "r", allow_pickle=True) as data:
                self.t = data["time_stamp"]
                self.v, self.s = data["EMAs"].tolist()
                self.cost_log = data["cost_log"].tolist()

                return True

        except FileNotFoundError:
            # returning False if the model failed to load
            return False

    # backward propagation
    def __get_cost(self, Y):
        m = Y.shape[1]

        # computing L2 regularization term
        L2_term = (self.hyperparams["lambd"] / (2 * m)) * np.sum(
            [np.sum(self.params["W" + str(l)] ** 2) for l in range(1, self.L + 1)])

        # computing cost
        cost = self.hyperparams["cost_function"]({"Y": Y, "Y_hat": self.output}) + L2_term

        return cost

    def __back_prop(self, Y):
        # resetting grads
        self.grads = {}

        # dZl is returned by the cost function
        dZL = self.hyperparams["cost_function"]({"Y": Y, "Y_hat": self.output, "back_prop": True})

        # using an identity function as the activation function gradient since we pass dZl in the place of dAL
        dA = self.__pass_backward(self.L, dZL, lambda dic: 1)

        # propagating backwards through the hidden layers
        for l in reversed(range(1, self.L)):
            dA = self.__pass_backward(l, dA, self.hyperparams["hidden_layer_activ_func"])

    def __pass_backward(self, l, dA, g_prime):
        m = dA.shape[1]

        # retrieving parameters
        W = self.params["W" + str(l)]

        # retrieving cached values
        cache = self.fore_prop_cache[l - 1]
        Z = cache["Z"]
        A_prev = cache["A_prev"]

        # drop out
        if "D" in cache:
            dA = self.__drop_out_back_prop(dA, cache["D"])

        # calculating gradients

        # dZ is returned by the activation function if back_prop is set to True
        dZ = g_prime({"input": Z, "back_prop": True}) * dA

        # Z = W * A_prev + b, dZ/db = 1
        db = np.mean(dZ, axis=1, keepdims=True)

        # checking if batch norm is enabled
        if self.hyperparams["batch_norm"] and l != self.L:
            # calculating dZ_orig and dgamma
            gamma = self.params["gamma" + str(l)]
            dZ, dgamma = self.__get_batch_norm_grads(dZ, gamma, cache)

            # caching dgamma
            self.grads["dgamma" + str(l)] = dgamma

        # L2 regularization term
        L2_term = self.hyperparams["lambd"] / m * W

        # Z = W * A_prev + b, dZ/dW = A_prev
        dW = 1 / m * np.dot(dZ, A_prev.T) + (self.hyperparams["lambd"] / m * W) + L2_term

        # Z = W * A_prev + b, dZ/dA_prev = W
        dA_prev = np.dot(W.T, dZ)

        # caching gradients
        self.grads["dW" + str(l)] = dW
        self.grads["db" + str(l)] = db

        return dA_prev

    @staticmethod
    def __get_batch_norm_grads(dZ_tilde, gamma, cache):
        # retrieving cached values
        Z_orig = cache["Z_orig"]
        Z_norm = cache["Z_norm"]
        mean = cache["mean"]
        std = cache["std"]

        m = Z_orig.shape[1]

        # Z_tilde = Z_norm * gamma + b, dZ_tilde/dgamma = Z_norm
        dgamma = np.mean(Z_norm * dZ_tilde, axis=1, keepdims=True)

        # Z_tilde = Z_norm * gamma + b, dZ_tilde/dZ_norm = gamma
        dZ_norm = gamma * dZ_tilde

        # computing dL_dmean
        dmean = np.sum(-1 / std * dZ_norm, axis=1, keepdims=True)

        # computing dL_dstd
        dstd = np.sum(-1 / std ** 2 * (Z_orig - mean) * dZ_norm, axis=1, keepdims=True)

        # computing gradients of mean and std in respect to Z
        dmean_dZ = 1 / m
        dstd_dZ = (Z_orig - mean) / (m * std)

        # adding all the parts to get the final gradient of Z
        dZ = dZ_norm / std + dmean * dmean_dZ + dstd * dstd_dZ

        return dZ, dgamma

    def __drop_out_back_prop(self, dA, D):
        # knocking neurons out
        dA = dA * D

        # correcting derivatives
        dA = dA / self.hyperparams["keep_prob"]

        return dA

    # optimization
    def __update_parameters(self):
        optimizer = self.hyperparams["optimizer"]

        # computing decayed learning rate
        decayed_learning_rate = self.__get_decayed_learning_rate()

        # looping through each parameter
        for param_key in self.params:
            grad_key = "d" + param_key

            # updating momentum and RMS
            v, s = self.__update_EMAs(grad_key)

            # updating parameters
            self.params[param_key] = optimizer({
                "param": self.params[param_key],
                "grad": self.grads[grad_key],
                "learning_rate": decayed_learning_rate,
                "v": v,
                "s": s
            })

    def __update_EMAs(self, key):
        beta1 = self.hyperparams["beta1"]
        beta2 = self.hyperparams["beta2"]

        # updating momentum
        self.v[key] = beta1 * self.v[key] + (1 - beta1) * self.grads[key]

        # updating root-mean-squared
        self.s[key] = beta2 * self.s[key] + (1 - beta2) * self.grads[key] ** 2

        # computing and returning v and s for optimization

        # correcting bias (bias should not be included in the EMAs)
        v = self.v[key] / (1 - beta1 ** self.t)
        s = self.s[key] / (1 - beta2 ** self.t)

        return v, s

    def __get_decayed_learning_rate(self):
        # computing decayed learning rate
        decayed_learning_rate = self.hyperparams["decay_function"]({
            "learning_rate": self.hyperparams["learning_rate"],
            "epoch": self.__get_current_epoch(),
            "decay_rate": self.hyperparams["decay_rate"],
            "decay_interval": self.hyperparams["decay_interval"]
        })

        return decayed_learning_rate

    def __get_current_epoch(self):
        # number_of_batches = numer_of_examples / batch_size
        batch_number = int(self.m / self.hyperparams["batch_size"])

        # epoch = t /
        epoch = int(self.t / batch_number)
        return epoch

    # debugging
    def grad_check(self, print_derivatives=False, epsilon=1e-7):
        dTheta_approx = np.array([])
        dTheta = np.array([])

        # reducing the dataset to three example
        X = self.X
        Y = self.Y

        X = X[:, 0: 3].reshape(X.shape[0], 3)
        Y = Y[:, 0: 3].reshape(Y.shape[0], 3)

        # computing analytical gradients
        self.fore_prop(X)
        self.__back_prop(Y)

        # looping over each elements of each parameters
        for p in self.params:
            for r in range(self.params[p].shape[0]):
                for c in range(self.params[p].shape[1]):
                    # computing numerical gradient
                    numerical_grad = self.__approx_derivative(X, Y, p, r, c, epsilon)

                    # saving results to a vector
                    analytical_grad = self.grads["d" + p][r][c]

                    dTheta_approx = np.append(dTheta_approx, numerical_grad)
                    dTheta = np.append(dTheta, analytical_grad)

                    # printing gradients and relative difference of each element if specified (off by default)
                    if print_derivatives:
                        difference = self.__get_relative_difference(analytical_grad, numerical_grad)

                        print("parameter: " + p)
                        print("numerical: " + str(numerical_grad))
                        print("analytical: " + str(self.grads["d" + p][r][c]))
                        print("error: " + str(difference))
                        print()

        # computing the relative difference between the numerical gradient and the analytical gradient
        difference = self.__get_relative_difference(dTheta, dTheta_approx)

        print("Error: " + str(difference))

    def __approx_derivative(self, X, Y, p, r, c, epsilon):
        # df/dx ≈ (f(x + h) - f(x - h)) / 2h

        # epsilon is equivalent to h in this case

        # equivalent to f(x + h)
        self.params[p][r][c] += epsilon
        self.fore_prop(X)
        J_plus = self.__get_cost(Y)

        # equivalent to f(x - h)
        self.params[p][r][c] -= 2 * epsilon
        self.fore_prop(X)
        J_minus = self.__get_cost(Y)

        # restoring the original value of the parameter
        self.params[p][r][c] += epsilon

        # approximating the derivative
        approx = (J_plus - J_minus) / (2 * epsilon)

        return approx

    @staticmethod
    def __get_relative_difference(dTheta, dTheta_approx):
        # relative difference = (a - b) / (a + b)
        # for vectors the magnitude of a - b over the magnitude of a + b
        numerator = np.linalg.norm(dTheta - dTheta_approx)
        denominator = np.linalg.norm(dTheta) + np.linalg.norm(dTheta_approx)
        difference = numerator / denominator

        return difference

    # execution
    def print_cost(self):
        # computing cost
        self.fore_prop(self.X)
        cost = self.__get_cost(self.Y)

        epoch = self.__get_current_epoch()

        # printing cost
        print("Epoch {0} -> Cost: {1}".format(epoch, cost))

        return cost

    def train(self, iterations, log_frequency=0):
        # reducing training set to mini batches
        batch_size = self.hyperparams["batch_size"]
        batch_number = int(self.m / batch_size)

        mini_batch_X = np.array_split(self.X, batch_number, axis=1)
        mini_batch_Y = np.array_split(self.Y, batch_number, axis=1)

        mini_batch_list = list(zip(mini_batch_X, mini_batch_Y))

        # repeating for "iterations" amount
        for e in range(iterations):
            # printing cost after a certain amount of iterations
            if e % log_frequency == 0:
                cost = self.print_cost()

                # logging cost
                self.cost_log.append(cost)

            # looping over each mini batch
            for mini_batch in mini_batch_list:
                # incrementing t
                self.t += 1

                # retrieving mini batch X and Y
                X = mini_batch[0]
                Y = mini_batch[1]

                # training the model
                self.fore_prop(X)
                self.__back_prop(Y)
                self.__update_parameters()

        self.print_cost()
