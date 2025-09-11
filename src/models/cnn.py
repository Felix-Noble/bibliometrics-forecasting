import tensorflow as tf
import keras_tuner as kt

def build_cnn(
    input_shape: tuple, 
    n_output: int,
    n_embeddings: int,
    decay_steps: int,
    
    max_conv_layers = 1,
    max_filters = 32,
    max_kernel_size_mult = 4,
    ):
    
    def build_model(hp):

        l2_regularization = hp.Float('l2_regularization', min_value=1e-5, max_value=1e-2, sampling='log')
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

        decay_rate = hp.Float('exp_decay_rate', min_value = 0.5, max_value = 1, sampling="linear")

        inputs = tf.keras.layers.Input(shape = (input_shape[0] * input_shape[1],))
        x = tf.keras.layers.Reshape(input_shape)(inputs)
        for i in range(hp.Choice("conv_layers", values=[1, max_conv_layers])):
            x = tf.keras.layers.Conv1D (
                filters = hp.Int("filters_" + str(i), 8, max_filters, step=8, default=16),
                kernel_size = hp.Int("kernel_size_" + str(i), n_embeddings*2, (max_kernel_size_mult)*n_embeddings, step=n_embeddings),
                #kernel_size = n_embeddings, 
                padding = "same",
                kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)

            )(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps = decay_steps,
            decay_rate = decay_rate
        ) 
        optimizer_name = hp.Choice("optimizer", ["adam"])
        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            optimizer, loss="binary_crossentropy", metrics=[tf.keras.metrics.Precision(), "accuracy"]
        )
        return model

    def build_largest_model():
        hp = kt.HyperParameters()
        # set parameter independent values (don't affect param count)
        hp.values["l2_regularization"] = 1e-2
        hp.values["learning_rate"] = 1e-2   
        hp.values["exp_decay_rate"] = 1.0

        hp.values['conv_layers'] = max_conv_layers
        for layer in range(hp.values['conv_layers']):
            hp.values[f'filters_{layer}'] = max_filters
            hp.values[f'kernel_size_{layer}'] = max_kernel_size_mult * n_embeddings

        hp.values['optimizer'] = "adam"

        model = build_model(hp)
        return model

    return build_model, build_largest_model


