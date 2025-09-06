from typing import Optional, Generator
import tensorflow as tf
from pathlib import Path
import os
import shutil

def process_tf_dataset(
        t_start: int,
        t_end: int,
        data_generator: Generator,
        x_reshape: Optional[tuple[int, ...]] = None,
        dataset_loc: str = "./data/tf/",
        overwrite: bool = True,
):

    if not os.path.exists(dataset_loc):
        os.makedirs(dataset_loc, exist_ok = True)

    for batch_data, batch_i in data_generator:
        print("processing batch", batch_i, "n_examples = ", batch_data.shape)
        x_cols = [col for col in batch_data.columns if "embedding" in str(col)]
        y_cols = [col for col in batch_data.columns if "label" in str(col)]
        dataset = tf.data.Dataset.from_tensor_slices( (batch_data[x_cols], batch_data[y_cols]), )

        if x_reshape is not None:
            def reshape_func(x, y):
                x = tf.reshape(x, x_reshape)
                return x, y
            dataset = dataset.map(reshape_func, num_parallel_calls = tf.data.AUTOTUNE)

        path = os.path.join(dataset_loc, f"{t_start}_{t_end}_{batch_i}")
        if os.path.exists(path):
            if overwrite:
                print(f"Overwriting dataset batch: {path}")
                shutil.rmtree(path)
            else:
                print("Skipping batch: {path}")
                continue
        tf.data.Dataset.save(dataset, path)


