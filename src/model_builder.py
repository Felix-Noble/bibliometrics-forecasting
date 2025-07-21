#src/build_model.py
import tensorflow as tf
import keras.layers as k_layers
import keras.regularizers as k_reg

def build_model(config):
    n_input = config["n_input"]
    n_output = config["n_output"]

    # Model definitions
    def conv_1DMLP(hp):
        
        l2_regularization = hp.Choice('l2_regularization', values=[0.01, 0.001, 0.0001])

        inputs = k_layers.Input(shape=n_input)
        x = inputs
        for i in range(hp.Int("conv_layers", 1, 3, default=1)):
            x = k_layers.Conv1D (
                filters = hp.Int("filters_" + str(i), 8, 64, step=8, default=16),
                kernel_size = hp.Int("kernel_size_" + str(i), 1149, 1915),
                activation = "relu",
                padding = "same",
                kernel_regularizer=k_reg.l2(l2_regularization)

            )(x)
            x = k_layers.AveragePooling1D()(x)
            x = k_layers.BatchNormalization()(x)
            x = k_layers.ReLU()(x)
        
        outputs = k_layers.layers.Dense(n_output, activation="softmax")(x)

        model = k_layers.Model(inputs, outputs)
        
        optimizer = hp.Choice("optimizer", ["adam"])

        model.compile(
            optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        return model
    


    # name -> model map
    model_name_map = {
        "conv_1DMLP" : conv_1DMLP,
                      }
    
    # Critical checks
    if config["model_name"] not in model_name_map.keys():
        return ValueError(f"Model name not recognised | recognised names : {model_name_map.keys()} ")
    
    return config["model_name"]

