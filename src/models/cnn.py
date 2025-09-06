import tensorflow as tf

def build_cnn(
    input_shape: tuple, 
    n_output: int,
    n_embeddings: int,
    decay_steps: int,
    ):
    
    def build_model(hp):

        l2_regularization = hp.Float('l2_regularization', min_value=1e-5, max_value=1e-2, sampling='log')
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

        decay_rate = hp.Float('exp_decay_rate', min_value = 0.5, max_value = 1, sampling="linear")

        inputs = tf.keras.layers.Input(shape = input_shape)
        #x = k_layers.Reshape((N_REF+1, n_embeddings))(inputs)
        x = inputs
        for i in range(hp.Choice("conv_layers", values=[1])):
            x = tf.keras.layers.Conv1D (
                filters = hp.Int("filters_" + str(i), 8, 64, step=8, default=16),
                kernel_size = hp.Int("kernel_size_" + str(i), n_embeddings*2, (5)*n_embeddings, step=n_embeddings),
                #kernel_size = n_embeddings, 
                padding = "same",
                kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)

            )(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(n_output, activation="softmax")(x)

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
            optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    return build_model


