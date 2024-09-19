from neural_net_module import trainer, algorithms
import h5py
import matplotlib.pyplot as plt

# loading datasets
with h5py.File("datasets/datasets.h5", "r") as trainingSet:
    X = trainingSet["dev_set_X"][:]
    Y = trainingSet["dev_set_Y"][:]

# initializing neuralnetwork
validation_model_trainer = trainer.Trainer(
    model_file=r"parameters/dev_model_final_params.npz",
    training_cache="parameters/dev_training_cache_final.npz"
)
validation_model_trainer.initialize_model({
    "layer_dims": [X.shape[0], 32, 1],
    "output_layer_activ_func": algorithms.Functions.sigmoid,
    "cost_function": algorithms.Functions.binary_cross_entropy,
    "batch_norm": True,
    "learning_rate": 0.015,
    "decay_function": algorithms.Functions.scheduled_decay,
    "decay_rate": 0.45,
    "decay_interval": 10,
    "lambd": 0.001,
    "keep_prob": 0.57
})

# loading dataset
validation_model_trainer.load_dataset(X, Y)

# training model

validation_model_trainer.train(100, log_frequency=10)

# saving model
validation_model_trainer.save_model()
validation_model_trainer.save_training_cache()

# plotting cost
plt.plot(validation_model_trainer.cost_log)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()
