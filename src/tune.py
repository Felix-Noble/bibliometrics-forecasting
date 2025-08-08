#src/fit.py
from functools import cache
import os
from os import wait
from src.utils.load_config import get_model_config, get_log_config, get_data_config, get_train_config
from src.utils.setup_logging import setup_logger
from pathlib import Path
import pandas as pd
import numpy as np
import psutil
# import dask
import pyarrow.compute as pc
import pyarrow.parquet as pq 
import pyarrow as pa

import ast
import time
import logging
import duckdb
import keras
import keras_tuner
import keras.layers as k_layers
import keras.regularizers as k_reg
import tensorflow as tf
import gc
# temp #

shuffle_logger = logging.getLogger("distributed.shuffle._scheduler_plugin")
# Set its level to ERROR to suppress warnings
shuffle_logger.setLevel(logging.ERROR)
# dask.config.set({'logging': {'distributed': logging.ERROR}})
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


def replace_ref_works(string_val):
    temp = ast.literal_eval(string_val)
    return [int(x.replace("https://openalex.org/W", "")) for x in temp]

def make_id_index_map(db_loc:str, id_col:str = "id_OpenAlex"):
    database = pq.ParquetDataset(db_loc)
    id_table = database.read(columns=[id_col])
    id_index_map = {int(id_val.as_py()): i for i, id_val in enumerate(id_table[id_col])}
    return id_index_map
def group_generator_pivot(db_dir: Path, main_df, id_index_map:dict, label_map:dict, n_back: int, feature_cols:list, batch_size:int =1024):
    main_db = pq.read_table(db_dir, memory_map=True)
    col_lookup_table = main_db.select(["id_OpenAlex"] + feature_cols) 
    all_ids = main_df['id_OpenAlex'].unique()
    logger.info(f"{all_ids.shape[0]} examples in split")
    for i in range(0, all_ids.shape[0], batch_size):
        batch_ids = all_ids[i:i + batch_size]
        df = main_df[main_df['id_OpenAlex'].isin(batch_ids)].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # 5. Rank references within each group and keep the top N
        df['ref_rank'] = df.groupby('id_OpenAlex').cumcount()
        df = df[df['ref_rank'] < n_back]
        ref_indicies = [id_index_map.get(id) for id in df["referenced_works_OpenAlex"]] 
        embeddings_ref = pc.take(col_lookup_table, pa.array(ref_indicies, type=pa.int64())).to_pandas()
        
        df = df.merge(embeddings_ref, on="id_OpenAlex", how="inner")
        
        #del embeddings_ref, ref_indicies
        
        features_df = df.pivot_table(
                index='id_OpenAlex',
                columns='ref_rank',
                values=feature_cols
            ).dropna()
            # Flatten the multi-level column index
        features_df.columns = [f'{col[0]}_{col[1]}' for col in features_df.columns]
        
        features_df = features_df[[f"embedding_{x}_{y}" for y in range(n_back) for x in range(len(feature_cols))]]
        self_indicies = [id_index_map.get(id) for id in features_df.index]
        self_embeddings = pc.take(col_lookup_table, pa.array(self_indicies, type=pa.int64())).to_pandas()
        
        # 7. Add embeddings for the source paper itself
        final_df = features_df.merge(
            self_embeddings,
            left_on="id_OpenAlex",
            right_index=True,
            how='inner',
            suffixes=('_ref', '_source')
        )
        yield (final_df.values, np.array([label_map.get(id) for id in final_df.index]))
    
def group_generator(db_dir: Path, df, id_index_map:dict, label_map:dict, n_back: int, feature_cols:list): 
    main_db = pq.read_table(db_dir, memory_map=True)
    col_lookup_table = main_db.select(feature_cols) 
    
    for group_id, group_df in df: 
        t1 = time.time()
        ref_works_icol = group_df.columns.get_loc("referenced_works_OpenAlex")
        indicies = [id_index_map.get(int(id)) for id in group_df.iloc[-n_back:, ref_works_icol]] + [id_index_map.get(group_id)]
        if not indicies:
            logger.error(f"No valid indicies found for id : {group_id}")
        embeddings = pc.take(col_lookup_table, pa.array(indicies, type=pa.int64())).to_pandas()
 
        embeddings = embeddings.values.reshape(-1,)
       # if np.isnan(embeddings).any() or np.isinf(embeddings).any():
       #     raise ValueError("Embedddings contain NaN or Inf") 
        yield (np.concatenate([embeddings], dtype=np.float32), label_map.get(group_id, -1))

def create_tf_dataset(batch_size, year, end_year, available_ids, id_index_map, db_files, sort_col, n_back, n_features=384, cache_dir = "./tf_cache/", step_name = None):
    cache_name = f"{step_name}{str(year)}-{str(end_year)}"
    output_signature = (
        tf.TensorSpec(shape=(None,(n_back+1) * n_features), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    if step_name is None:
        raise ValueError("Must give step name")
    if not list((Path(cache_dir).glob(f"{cache_name}*"))):
        logger.info(f"Cache not found for {step_name}, generating data")
        df = duckdb.sql(f"""SELECT id_OpenAlex, publication_date_int, referenced_works_OpenAlex 
        FROM read_parquet({db_files}) 
        WHERE publication_date_int >= {year} AND publication_date_int < {end_year}""").df()

        df = df.sort_values(sort_col, ascending=True)

        df["referenced_works_OpenAlex"] = df["referenced_works_OpenAlex"].apply(replace_ref_works)
        available_ids.columns = ["referenced_works_OpenAlex"] 
        df = df.explode("referenced_works_OpenAlex")
        df = df.merge(available_ids, on="referenced_works_OpenAlex", how="inner")
        
       # df = df.groupby("id_OpenAlex")
        embedding_cols = [f"embedding_{x}" for x in range(n_features)]
        y_df = duckdb.sql(f"""
        SELECT id_OpenAlex, higher_than_median_year 
        FROM read_parquet({db_files}) 
        WHERE publication_date_int >= {year} AND publication_date_int < {end_year}
        """).df()
        label_map = pd.Series(y_df.higher_than_median_year.values, index=y_df.id_OpenAlex).to_dict()
        del y_df

        dataset = tf.data.Dataset.from_generator(
        lambda : group_generator_pivot(Path(db_files[0]).parent, df, id_index_map, label_map, n_back, embedding_cols, batch_size=(1024*2)),
            output_signature=output_signature,
        )
        """
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
            ((n_back+1) * n_features),       # Shape for 1D feature vector (pad the variable dimension)
            (),          # Shape for scalar label (it's empty)
            ),
            padding_values=(
            tf.constant(0, dtype=tf.float32),
            tf.constant(0, dtype=tf.int32), 
            )
        )
        """
        
        dataset = dataset.cache(os.path.join(cache_dir, cache_name))
        for _ in dataset:
            pass

        logger.info(f"Dataset {step_name} cached")

    else:
        dataset = tf.data.Dataset.from_generator(
        lambda : None,
            output_signature=output_signature
        )
        dataset = dataset.cache(os.path.join(cache_dir, cache_name))
        
        logger.info(f"Dataset {step_name} loaded")

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
   
    return dataset        


logger = logging.getLogger(Path(__file__).stem)

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

    database_loc = data_config["database_loc"]

    start_year = train_config["start_year"]
    end_year = train_config["end_year"]
    CV_delta = train_config["CV_delta"]
    val_size = train_config["val_size"]
    test_size = train_config["test_size"]

    import os
    database_files = [str(x) for x in Path(database_loc).glob("*.parquet")]
    available_ids = duckdb.sql(f"SELECT id_OpenAlex FROM read_parquet({database_files})").df()
    id_index_map = make_id_index_map(database_loc, "id_OpenAlex")

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
                filters = hp.Int("filters_" + str(i), 16, 64, step=16, default=16),
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

        val = create_tf_dataset(batch_size, val_start, test_start, available_ids, id_index_map, database_files, sort_col, N_REF, n_embeddings, step_name="val")
        train = create_tf_dataset(batch_size, start_year_int, val_start, available_ids, id_index_map, database_files, sort_col, N_REF, n_embeddings, step_name="train")

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

        test = create_tf_dataset(batch_size, test_start, test_start + test_size_int, available_ids, id_index_map, database_files, sort_col, N_REF, n_embeddings, step_name="test")
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
