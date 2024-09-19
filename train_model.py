from neural_net_module import trainer, algorithms
import h5py
import matplotlib.pyplot as plt

# loading datasets4
with h5py.File("datasets/datasets.h5", "r") as trainingSet:
    X = trainingSet["training_set_X"][:]
    Y = trainingSet["training_set_Y"][:]

# initializing neuralnetwork
neuralnet_trainer = trainer.Trainer(
    model_file=r"parameters/final_model_params.npz",
    training_cache="parameters/final_model_training_cache.npz"
)
neuralnet_trainer.initialize_model({
    "layer_dims": [X.shape[0], 32, 1],
    "output_layer_activ_func": algorithms.Functions.sigmoid,
    "cost_function": algorithms.Functions.binary_cross_entropy,
    "batch_norm": True,
    "batch_size": 512,
    "learning_rate": 0.015,
    "decay_function": algorithms.Functions.scheduled_decay,
    "decay_rate": 0.45,
    "decay_interval": 10,
    "lambd": 0.5,
    "keep_prob": 0.5
})

# loading dataset
neuralnet_trainer.load_dataset(X, Y)

# training model
neuralnet_trainer.train(100, log_frequency=10)

# saving model
neuralnet_trainer.save_model()
neuralnet_trainer.save_training_cache()

# plotting cost
plt.plot(neuralnet_trainer.cost_log)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()
