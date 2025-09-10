#src/tune.py
from src.utils.load_config import get_model_config, get_log_config, get_data_config, get_train_config
from src.utils.setup_logging import setup_logger
from src.load_data import create_tf_dataset
from src.models.registry import get_model
from sklearn.metrics import balanced_accuracy_score, precision_recall_curve, precision_score, recall_score
from pathlib import Path
import pandas as pd
import numpy as np
import psutil
import logging
import keras_tuner
import tensorflow as tf
import math
import json
import os

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
        if best_score is None:
            is_unpromising = False
        elif self.objective_direction == 'max': # e.g., for accuracy
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

def select_files(data_dir: Path, t_min: int, t_max: int):
    candidates = Path(data_dir).glob("*_*_shard_*.tfrecord")
    valid_files = []
    for f in candidates:
        f_min, f_max = map(int, f.stem.split("_")[:2])
        if f_max >= t_min and f_min <= t_max:
            valid_files.append(str(f))
    return valid_files

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
    
    tf_dir = data_config["tf_dataset_loc"]
    n_features = data_config["n_features"]
    y_cols = data_config["y_cols"]
     
    model_name = model_config["model_name"]
    n_output = 2
    #embedding_cols = [f"embedding_{x}" for x in range(n_embeddings)]
    max_timepoints = data_config["max_timepoints"]
     
    n_input = n_features * (max_timepoints + 1)
    tf.random.set_seed(2025)
   
    max_trials = train_config["max_trials"]
    ratio_initial_points = train_config["ratio_initial_points"]
    patience = train_config["patience"]
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
      
    mem = psutil.virtual_memory()
    available = mem.available 
    example_size = n_input * np.float32().nbytes
    example_size = example_size * 16
    batch_size = available // example_size

    start_year_int = start_year.value
    test_size_int = test_size.value
    val_size_int = val_size.value
    
    feature_description = {
        'features':tf.io.FixedLenFeature([n_input], tf.float32),
        'target': tf.io.FixedLenFeature([len(y_cols)], tf.float32),
    }
    def parse_example(serialized_example):
        parsed = tf.io.parse_single_example(serialized_example, feature_description)
        return parsed["features"], parsed["target"]

    def build_dataset(files, batch_size=128, shuffle=True):
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
        ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=10_000)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

   
    tuner_search = True
    for current_slice_end_year in range(first_slice_end_year, end_year, CV_delta):
        previous_slice_end_year = start_year # reset prev slice value for entire dataset load every time
        # initialise tuner # 
        # for current_slice_end_year in pd.timedelta_range(first_slice_end_year, end_year):
        logger.info(f"Importing slice from temporal range: {previous_slice_end_year.year} to {current_slice_end_year}")
        ## select range ##
        greater_than = pd.to_datetime(previous_slice_end_year, format="%Y").value
        less_than = pd.to_datetime(current_slice_end_year, format="%Y").value
        test_start: int = pd.to_datetime(f"{current_slice_end_year}", format="%Y").value - test_size_int
        val_start: int = test_start - val_size_int
        
        logger.info(f"Processing examples of size {example_size / (1024 * 1024)} MB | Batch size = {batch_size}")
        train_files = select_files(tf_dir, start_year_int, val_start)
        val_files = select_files(tf_dir, val_start, test_start)

        train_ds = build_dataset(train_files, batch_size = batch_size)
        val_ds = build_dataset(val_files, batch_size = batch_size)
        with open(os.path.join(tf_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        n_train_examps = sum([metadata["./" + file] for file in train_files]) 
        n_train_examps = 90e3
        decay_steps = math.ceil(batch_size / n_train_examps)

        model_builder = get_model(model_name, input_shape=(max_timepoints+1, n_features), n_output=n_output, n_embeddings=n_features, decay_steps=decay_steps)

        if tuner_search:
            tuner = keras_tuner.BayesianOptimization(
                hypermodel = model_builder,
                objective=keras_tuner.Objective("val_precision", direction = "max"),
                max_trials=max_trials,
                num_initial_points=max_trials // ratio_initial_points,
                executions_per_trial=2,
                overwrite=True,
                directory=f"./results_dir/{start_year.year}-{current_slice_end_year}",
                project_name=model_name,
            )
            early_stop_val = tf.keras.callbacks.EarlyStopping(
                monitor='val_precision',
                patience=patience,
                mode='max'
            )
            early_stop_best = StopIfUnpromisingTrial(
                tuner=tuner,
                patience=15,
                score_threshold_ratio=0.95,
            )


            logger.info("Starting search...")
            tuner.search(train_ds, epochs=100, validation_data=val_ds, callbacks=[early_stop_val, early_stop_best])
            models = tuner.get_best_models(num_models=1)
            best_model: tf.keras.Model = models[0]
            best_model.summary()
            best_model.save(f"./data/{model_name}-best_model.h5")
            logger.info("Hyperparameter search complete. Starting final training process.")
            tuner_search = False
        
        else:
            #best_model: tf.keras.Model = tf.keras.models.load_model(f"./data/{model_name}-best_model.h5")
            hp = keras_tuner.HyperParameters()
            hp.Fixed("l2_regularisation", 0.0039705)
            hp.Fixed("learning_rate", 2.8242e-5)
            hp.Fixed("exp_decay_rate", 0.9964)
            hp.Fixed("conv_layers", 1)
            hp.Fixed("filters_0", 32)
            hp.Fixed("kernel_size_0", 768)
            hp.Fixed("optimizer", "adam")
            best_model = model_builder(hp) 
            
        final_training_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_precision',      
            patience=5,                 
            verbose=1,
            mode='max',                
            restore_best_weights=True  
        )

        logger.info("Starting final training of the best model...")

        best_model.fit(
            train_ds,
            epochs=100,  
            validation_data=val_ds,
            callbacks=[final_training_callback]
        )

        logger.info("Final training complete.")
        
        test_files = select_files(tf_dir, test_start,test_start + test_size_int)
        test_ds = build_dataset(test_files, batch_size = batch_size)

        logger.info("Evaluating final model on the test set...")
        test_loss, test_accuracy = best_model.evaluate(test_ds)

        y_pred_prob = best_model.predict(test_ds)
        y_true = np.concatenate([y for x, y in test], axis=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        logger.info(f"Accuracy on the unseen test set: {test_accuracy:.4f}")    

        metrics = pd.DataFrame(
        [[test_loss, test_accuracy, balanced_accuracy_score(y_true, y_pred), precision_recall_curve(y_true, y_pred), recall_score(y_true, y_pred), precision_recall_curve(y_true, y_pred)]], 
columns=["loss", "accuracy", "balanced_accuracy", "precision", "recall", "precision_recall_curve"]
        )
        metrics.to_csv(f"./test_results_dir/{current_slice_end_year - train_config['test_size']}-{current_slice_end_year}.csv")
        print(metrics)
if __name__ == "__main__":
    # main()
    pass
