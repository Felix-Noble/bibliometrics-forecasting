#src/fit.py
from src.utils.load_config import get_model_config, get_log_config, get_data_config, get_train_config
from src.utils.setup_logging import setup_logger
from pathlib import Path
import pandas as pd
import numpy as np
import psutil
import logging
import keras
import keras_tuner
import keras.layers as k_layers
import keras.regularizers as k_reg
import tensorflow as tf

logger = logging.getLogger(Path(__file__).stem)

class StopIfUnpromisingTrial(tf.keras.callbacks.Callback):
    """
    A custom Keras callback to stop a trial early if it's not beating the
    best score from previous trials after a certain number of epochs.
    """
    def __init__(self, tuner, patience=15, score_threshold_ratio=0.95):
        super().__init__()
        self.tuner = tuner
        self.patience = patience
        self.score_threshold_ratio = score_threshold_ratio

        self.wait = 0
        # Determine if we want to maximize or minimize the metric
        self.objective_direction = self.tuner.oracle.objective.direction 
    def on_epoch_end(self, epoch, logs=None):
        # Don't do anything for the first few epochs to give the model a chance
        if epoch < self.patience:
            return

        # Get the current score for this trial
        current_score = logs.get(self.tuner.oracle.objective.name)
        if current_score is None:
            return # Metric not available

        # Get the best score from all previous trials
        best_trials = self.tuner.oracle.get_best_trials(1)
        if not best_trials:
            return # No best trial yet (we are in the first trial)

        best_score = best_trials[0].score 

        # Compare scores based on the objective (e.g., 'max' or 'min')
        is_unpromising = False
        if self.objective_direction == 'max': # e.g., for accuracy
            if current_score < best_score * self.score_threshold_ratio:
                is_unpromising = True
        else: # 'min', e.g., for loss
            if current_score > best_score * (1/self.score_threshold_ratio):
                is_unpromising = True
        
        # If the trial is unpromising, stop it
        if is_unpromising:
            print(f"\nTrial is unpromising. Current score ({current_score:.4f}) is not "
                  f"beating the best score ({best_score:.4f}). Stopping trial.")
            self.model.stop_training = True


def main():
    setup_logger(logger, get_log_config())
    gpus = tf.config.list_physical_devices()
    if gpus:
        logger.info(f"GPUs found: {len(gpus)}")
    else:
        logger.error(f"No GPUs found")

    data_config = get_data_config()
    model_config = get_model_config()
    train_config = get_train_config()

    n_embeddings = data_config["Nembeddings"]
    embedding_cols = [f"embedding_{x}" for x in range(n_embeddings)]
    N_REF = 5
    n_input = n_embeddings * (N_REF + 1)
    n_output = 2
    tf.random.set_seed(2025)

    db_dir = data_config["database_loc"]

    start_year = train_config["start_year"]
    end_year = train_config["end_year"]
    CV_delta = train_config["CV_delta"]
    val_size = train_config["val_size"]
    test_size = train_config["test_size"]

    first_slice_end_year = start_year + train_config["start_train_size"] + val_size + test_size

    start_year = pd.to_datetime(f"{start_year}", format="%Y")
    val_size = pd.Timedelta(train_config["val_size"] * 52, "W")
    test_size = pd.Timedelta(train_config["test_size"] * 52, "W")

    previous_slice_end_year = start_year

    # keras_model = define_model(model_config)
    def build_model(hp):
        l2_regularization = hp.Float('l2_regularization', min_value=1e-5, max_value=1e-2, sampling='log')
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        decay_rate = hp.Choice('exp_decay_rate', values=[1.0])

        inputs = k_layers.Input(shape=(n_input))
        x = k_layers.Reshape((N_REF+1, n_embeddings))(inputs)
        for i in range(hp.Int("conv_layers", 1, 3, step=1, default=1)):
            x = k_layers.Conv1D (
                filters = hp.Int("filters_" + str(i), 8, 64, step=8, default=16),
                kernel_size = hp.Int("kernel_size_" + str(i), n_embeddings*2, (N_REF)*n_embeddings, step=n_embeddings),
                
                padding = "same",
                kernel_regularizer=k_reg.l2(l2_regularization)

            )(x)

            x = k_layers.BatchNormalization()(x)
            x = k_layers.ReLU()(x)
        x = k_layers.Flatten()(x)
        outputs = k_layers.Dense(n_output, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=30, # Note: This needs to be defined
        decay_rate=decay_rate
        ) 
        optimizer_name = hp.Choice("optimizer", ["adam"])
        if optimizer_name == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        return model


    start_year_int = start_year.value
    test_size_int = test_size.value
    val_size_int = val_size.value
    logger.info(f"{start_year_int} | {test_size_int} | {val_size_int}")
    for current_slice_end_year in range(first_slice_end_year, end_year, CV_delta):
        previous_slice_end_year = start_year # reset prev slice value for entire dataset load every time
        # initialise tuner # 
        tuner = keras_tuner.BayesianOptimization(
            hypermodel=build_model,
            objective="val_accuracy",
            max_trials=25,
            num_initial_points=7,
            executions_per_trial=2,
            overwrite=True,
            directory=f"./results_dir/{start_year.year}-{current_slice_end_year}",
            project_name=model_config["model_name"],
        )
        early_stop_val = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            mode='max'
        )
        early_stop_best = StopIfUnpromisingTrial(
            tuner=tuner,
            patience=15,
            score_threshold_ratio=0.95,
        )

        # for current_slice_end_year in pd.timedelta_range(first_slice_end_year, end_year):
        logger.info(f"Importing slice from temporal range: {previous_slice_end_year.year} to {current_slice_end_year}")
        ## select range ##
        greater_than = pd.to_datetime(previous_slice_end_year, format="%Y").value
        less_than = pd.to_datetime(current_slice_end_year, format="%Y").value
        test_start: int = pd.to_datetime(f"{current_slice_end_year}", format="%Y").value - test_size_int
        val_start: int = test_start - val_size_int

        N_REFS = 10
        N_OUT = 1
        y_cols = ["higher_than_median_year"]
        sort_col = "publication_date_int"       
        mem = psutil.virtual_memory()
        available = mem.available
        available = 4500 * (1024 * 1024)
        example_size = (N_REF+1) * n_embeddings * np.float32().nbytes
        batch_size = available // example_size
        logger.info(f"Processing examples of size {example_size / (1024 * 1024)} MB in {batch_size} sized batches")

        val = create_tf_dataset(db_dir=db_dir, year=val_start, end_year=test_start, sort_col=sort_col, n_back=N_REF, n_features=n_embeddings, step_name="val")
        train = create_tf_dataset(db_dir=db_dir, year=start_year_int, end_year=val_start, sort_col=sort_col, n_back=N_REF, n_features=n_embeddings, step_name="train")

        logger.info("Starting search...")
        tuner.search(train, epochs=100, validation_data=val, callbacks=[early_stop_val, early_stop_best])
        models = tuner.get_best_models(num_models=2)
        best_model = models[0]
        best_model.summary()
        
        logger.info("Hyperparameter search complete. Starting final training process.")

        # 1. Get the best hyperparameters found by the tuner.
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Best hyperparameters: {best_hps.values}")

        # 2. Build a fresh instance of the model with these best hyperparameters.
        final_model = tuner.hypermodel.build(best_hps)
        logger.info("Built best model for final training.")

        # 3. Create a new EarlyStopping callback for the final training phase.
        #    It will stop training when validation loss no longer improves and
        #    crucially, restore the model weights from the best epoch.
        final_training_callback = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',         # Monitor loss on the validation set
            patience=5,                 # Stop if val_loss doesn't improve for 5 epochs
            verbose=1,
            mode='max',                 # We want to minimize the loss
            restore_best_weights=True   # CRITICAL: This ensures the model returned is the one with the lowest val_loss
        )

        logger.info("Starting final training of the best model...")

        # 4. Retrain the model on the full training data.
        #    Set a high number of epochs; EarlyStopping will find the optimal number automatically.
        final_model.fit(
            train,
            epochs=100,  # Set a high number; EarlyStopping will handle stopping it.
            validation_data=val,
            callbacks=[final_training_callback]
        )

        logger.info("Final training complete.")

        # 5. Create the unseen test dataset.
        logger.info("Creating the test dataset for final evaluation...")

        test = create_tf_dataset(db_dir=db_dir, year=test_start, end_year=test_start + test_size_int, sort_col=sort_col, n_back=N_REF, n_features=n_embeddings, step_name="test")
        # 6. Evaluate the final, best model on the test set.
        logger.info("Evaluating final model on the test set...")
        test_loss, test_accuracy = final_model.evaluate(test)

        # 7. Store and print the result.
        logger.info(f"Accuracy on the unseen test set: {test_accuracy:.4f}")    
        pd.DataFrame([[test_loss, test_accuracy]], columns=["loss", "accuracy"]).to_csv(f"./test_results_dir/{start_year.year}-{current_slice_end_year}.csv")
    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # # Start a new MLflow run to store the final, best model and its full history.
    # with mlflow.start_run(run_name="Best_Model_Training"):

#     # Log the best hyperparameters
#     mlflow.log_params(best_hps.values)
#     print(f"Logged Best Hyperparameters: {best_hps.values}")

#     # Build the model with the best hyperparameters
#     best_model = tuner.hypermodel.build(best_hps)

#     # Retrain the model on the full dataset to get final training history
#     history = best_model.fit(
#         x_train, y_train,
#         epochs=50, # Use the desired number of epochs for the final model
#         validation_split=0.2
#     )

#     # Log the full training history (metrics per epoch)
#     for epoch in range(len(history.history['loss'])):
#         metrics_to_log = {
#             'train_loss': history.history['loss'][epoch],
#             'train_accuracy': history.history['accuracy'][epoch],
#             'val_loss': history.history['val_loss'][epoch],
#             'val_accuracy': history.history['val_accuracy'][epoch]
#         }
#         mlflow.log_metrics(metrics_to_log, step=epoch)

#     print("Logged full training history for the best model.")

#     # Log the final model to MLflow
#     mlflow.keras.log_model(best_model, "best-model")
#     print("Best model has been logged to MLflow.")

#     # You can also log other artifacts, like a summary of the best model
#     summary_path = "best_model_summary.txt"
#     with open(summary_path, "w") as f:
#         best_model.summary(print_fn=lambda x: f.write(x + '\n'))
#     mlflow.log_artifact(summary_path)
#     print(f"Logged model summary to {summary_path}.")

# print("\nMLflow logging complete. Check the MLflow UI for results.")
# 
# tuner.search_space_summary()

if __name__ == "__main__":
    # main()
    pass
