#src/tune.py
from pandas._libs.hashtable import mode
from src.utils.load_config import get_model_config, get_log_config, get_data_config, get_train_config
from src.utils.setup_logging import setup_logger
from src.models.registry import get_model
from src.export_tf_dataset import tf_dataset_from_generator, tf_dataset_from_slices, build_optimized_tfrecord_dataset
from src.callbacks.custom_prog_bar import BatchTimeLogger
from sklearn.metrics import balanced_accuracy_score, precision_recall_curve, precision_score, recall_score
from pathlib import Path
import pandas as pd
import numpy as np
import psutil
import logging
import keras_tuner
import tensorflow as tf
from tensorflow.keras.callbacks import ProgbarLogger
import math
import json
import os
import time
import gc
logger = logging.getLogger(Path(__file__).stem)
setup_logger(logger, get_log_config())

class StopIfUnpromisingTrial(tf.keras.callbacks.Callback):
    """
    A custom Keras callback to stop a trial early if it's not beating the
    best score from previous trials after a certain number of epochs.
    """
    def __init__(self, tuner, patience=7, score_threshold_ratio=0.97):
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

def configure_gpu(allow_memory_growth = True):
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid taking all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, allow_memory_growth)
            
            # Optional: Set mixed precision for better performance
            #tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            logger.info(f"Configured {len(gpus)} GPU(s) with memory growth")
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
    else:
        logger.warning("No GPUs found - training will use CPU")

def select_files(data_dir: Path, t_min: int, t_max: int):
    candidates = Path(data_dir).glob("*_*_shard_*.tfrecord")
    valid_files = []
    for f in candidates:
        f_min, f_max = map(int, f.stem.split("_")[:2])
        if f_max >= t_min and f_min <= t_max:
            valid_files.append(str(f))
    return valid_files

def find_max_batch_size_empirically(
    model_builder_fn,  # Should return COMPILED model
    input_shape: tuple,
    max_batch_size: int = 1024*5,  # Start reasonable
    min_batch_size: int = 32,
    test_steps: int = 5,       # Fixed number of steps
    safety_factor: float = 0.8,
    verbose: bool = True
    ):
    """Test with compiled model but synthetic data"""

    def test_batch_size(batch_size: int) -> bool:
        try:
            tf.keras.backend.clear_session()

            if verbose:
                print(f"Testing batch size: {batch_size}")

                # Build and compile model (this is correct!)
                model = model_builder_fn()

                # Create synthetic data for EXACT number of test steps
                total_samples = batch_size * test_steps
                test_input = tf.random.normal((total_samples, *input_shape))
                test_labels = tf.random.uniform((total_samples, 1), maxval=2, dtype=tf.int32)
                test_labels = tf.cast(test_labels, tf.float32)

                # This will always be exactly `test_steps` steps
                dataset = tf.data.Dataset.from_tensor_slices((test_input, test_labels))
                dataset = dataset.batch(batch_size)

                if verbose:
                    print(f"  Running exactly {test_steps} steps with batch size {batch_size}")

                # Train for exactly test_steps (should show consistent X/X steps)
                model.fit(dataset, epochs=1, verbose=1 if verbose else 0)

                return True

        except tf.errors.ResourceExhaustedError:
            if verbose:
                print(f"  ✗ OOM at batch size {batch_size}")
                return False
        finally:
            tf.keras.backend.clear_session()    
    # Binary search for maximum working batch size
    low = min_batch_size
    high = max_batch_size
    best_batch_size = min_batch_size

    if verbose:
        print(f"Starting binary search between {low} and {high}")

    while low <= high:
        mid = (low + high) // 2

        if test_batch_size(mid):
            best_batch_size = mid
            low = mid + 1
            if verbose:
                print(f"  ✓ Batch size {mid} works, trying larger")
        else:
            high = mid - 1
            if verbose:
                print(f"  ✗ Batch size {mid} failed, trying smaller")

    # Apply safety factor
    safe_batch_size = max(min_batch_size, int(best_batch_size * safety_factor))

    if verbose:
        print(f"\nEmpirical results:")
        print(f"  Maximum working batch size: {best_batch_size}")
        print(f"  Recommended safe batch size: {safe_batch_size} (with {safety_factor:.1%} safety factor)")

    tf.keras.backend.clear_session()
    return safe_batch_size

def main():
    configure_gpu() 

    data_config = get_data_config()
    model_config = get_model_config()
    train_config = get_train_config()
    
    sys_GRAM_MiB = train_config["GRAM_MiB"]

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

    #_, build_largest_model = get_model(model_name, input_shape=(max_timepoints+1, n_features), n_output=n_output, n_embeddings=n_features, decay_steps=1)

    batch_size = 512
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
        dataset = (
            tf.data.Dataset.from_tensor_slices(files)
            .shuffle(len(files))  # shuffle file order for good mixing
            .interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=8,  # how many files to read from in parallel
                num_parallel_calls=tf.data.AUTOTUNE
            )
            .map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)  # or your batch size
            .prefetch(20)
        )
        return dataset
       
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
        
        train_files = select_files(tf_dir, start_year_int, val_start)
        val_files = select_files(tf_dir, val_start, test_start)

        train_ds = build_optimized_tfrecord_dataset(train_files, parse_example, batch_size=batch_size)
        val_ds = build_optimized_tfrecord_dataset(val_files, parse_example, batch_size=batch_size)
        #train_ds = tf_dataset_from_generator(start_year_int, val_start)
        #val_ds = tf_dataset_from_generator(val_start, test_start)
        #test_ds = tf_dataset_from_generator(test_start, test_start + test_size_int)

        #train_ds = tf_dataset_from_slices(start_year_int, val_start)
        #val_ds = tf_dataset_from_slices(val_start, test_start)

        with open(os.path.join(tf_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        n_train_examps = sum([metadata["n_examples"]["./" + file] for file in train_files]) 
        n_steps_per_epoch = math.ceil(n_train_examps / batch_size)
        model_builder, _ = get_model(model_name, input_shape=(max_timepoints+1, n_features), n_output=n_output, n_embeddings=n_features, decay_steps=n_steps_per_epoch)
        
        logger.info(f"Processing {n_train_examps} train examples in {n_steps_per_epoch} steps | Batch size = {batch_size}")
        if tuner_search:
            tuner = keras_tuner.BayesianOptimization(
                hypermodel = model_builder,
                objective=keras_tuner.Objective("val_precision", direction = "max"),
                max_trials=max_trials,
                num_initial_points=max_trials // ratio_initial_points,
                executions_per_trial=1,
                overwrite=True,
                directory=f"./results_dir/{start_year.year}-{current_slice_end_year}",
                project_name=model_name,
            )
            early_stop_val = tf.keras.callbacks.EarlyStopping(
                monitor='val_precision',
                patience=patience,
                mode='max'
            )
            best_stop_val = StopIfUnpromisingTrial(tuner)
            logger.info("Starting search...")

            tuner.search(train_ds, 
                         epochs = 100, 
                         #steps_per_epoch = n_steps_per_epoch,
                         validation_data = val_ds, 
                         validation_freq = 1,
                         verbose = 1,
                         callbacks = [early_stop_val, best_stop_val],
                         )
            models = tuner.get_best_models(num_models=1)
            best_model: tf.keras.Model = models[0]
            best_model.summary()
            best_model.save(f"./data/{model_name}-best_model.h5")
            logger.info("Hyperparameter search complete. Starting final training process.")
            tuner_search = False
        
        else:
            best_model: tf.keras.Model = tf.keras.models.load_model(f"./data/{model_name}-best_model.h5")

        logger.info("Starting final training of the best model...")

        best_model.fit(
            train_ds,
            epochs=100,  
            validation_data=val_ds,
            callbacks=[early_stop_val]
        )

        logger.info("Final training complete.")
        
        test_files = select_files(tf_dir, test_start,test_start + test_size_int)
        test_ds = build_optimized_tfrecord_dataset(test_files, parse_example)

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

def test():
    gpus = tf.config.list_physical_devices()
    logger.info(gpus)
    if gpus:
        logger.info(f"GPUs found: {len(gpus)}")
    else:
        logger.error(f"No GPUs found")

if __name__ == "__main__":
    pass
