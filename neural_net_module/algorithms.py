import numpy as np


class Functions:
    @staticmethod
    def relu(dic):
        dic = {**{"back_prop": False}, **dic}

        Z = dic["input"]
        back_prop = dic["back_prop"]

        if back_prop:
            # returning the derivative if back_prop is on
            return np.where(Z < 0, 0, 1)

        return np.maximum(Z, 0)

    @staticmethod
    def leaky_relu(dic):
        # leak is set to 0.01 for default
        dic = {**{"leak": 0.01, "back_prop": False}, **dic}

        Z = dic["input"]
        back_prop = dic["back_prop"]
        leak = dic["leak"]

        if back_prop:
            # returning the derivative if back_prop is on
            return np.where(Z < 0, leak, 1)

        return np.maximum(Z, Z * leak)

    @staticmethod
    def sigmoid(dic):
        dic = {**{"back_prop": False}, **dic}

        Z = dic["input"]
        back_prop = dic["back_prop"]

        if back_prop:
            # returning the derivative if back_prop is on
            sigmoid_Z = Functions.sigmoid({"input": Z})
            return np.multiply(sigmoid_Z, 1 - sigmoid_Z)

        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def softmax(dic):
        Z = dic["input"]

        # Z max is subtracted for numerically stability (this does not affect the gradient)
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)

        t = np.exp(Z_shifted)
        t_sum = np.sum(t, axis=0, keepdims=True)

        return t / t_sum

    # cost functions
    @staticmethod
    def binary_cross_entropy(dic):
        dic = {**{"back_prop": False, "epsilon": 1e-8}, **dic}

        Y = dic["Y"]
        Y_hat = dic["Y_hat"]
        back_prop = dic["back_prop"]
        epsilon = dic["epsilon"]

        if back_prop:
            # returning the chained gradient of bce and sigmoid
            return Y_hat - Y

        return -np.mean(np.sum(Y * np.log(Y_hat + epsilon) + (1 - Y) * np.log(1 - Y_hat + epsilon), axis=0))

    @staticmethod
    def categorical_cross_entropy(dic):
        dic = {**{"back_prop": False, "epsilon": 1e-8}, **dic}

        Y = dic["Y"]
        Y_hat = dic["Y_hat"]
        epsilon = dic["epsilon"]
        back_prop = dic["back_prop"]

        if back_prop:
            # returning the chained gradient of cce and softmax
            return Y_hat - Y

        return -np.mean(np.sum(Y * np.log(Y_hat + epsilon), axis=0))

    # initialization methods
    @staticmethod
    def he(dic):
        n = dic["n"]
        n_prev = dic["n_prev"]

        return np.random.randn(n, n_prev) * np.sqrt(2 / n_prev)

    @staticmethod
    def xavier(dic):
        n = dic["n"]
        n_prev = dic["n_prev"]

        return np.random.randn(n, n_prev) * np.sqrt(1 / n_prev)

    # optimization methods
    @staticmethod
    def gradient_decent(dic):
        param = dic["param"]
        grad = dic["grad"]
        learning_rate = dic["learning_rate"]

        return param - learning_rate * grad

    @staticmethod
    def momentum(dic):
        param = dic["param"]
        learning_rate = dic["learning_rate"]
        v = dic["v"]

        return param - learning_rate * v

    @staticmethod
    def RMSprop(dic):
        dic = {**{"epsilon": 1e-8}, **dic}

        param = dic["param"]
        grad = dic["grad"]
        learning_rate = dic["learning_rate"]
        s = dic["s"]
        epsilon = dic["epsilon"]

        return param - (learning_rate * grad / (np.sqrt(s) + epsilon))

    @staticmethod
    def adam(dic):
        dic = {**{"epsilon": 1e-8}, **dic}

        param = dic["param"]
        learning_rate = dic["learning_rate"]
        v = dic["v"]
        s = dic["s"]
        epsilon = dic["epsilon"]

        return param - (learning_rate * v / (np.sqrt(s) + epsilon))

    # learning rate decay function
    @staticmethod
    def scheduled_decay(dic):
        learning_rate = dic["learning_rate"]
        epoch = dic["epoch"]
        decay_rate = dic["decay_rate"]
        interval = dic["decay_interval"]

        return learning_rate / (1 + decay_rate * np.floor(epoch / interval))
