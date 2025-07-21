import tensorflow 
import numpy as np
import pandas as pd
import duckdb
import dask.dataframe as dd
from dask.distributed import Client
from dask_sql import Context
import keras 
import keras_tuner
import tensorflow as tf
from utils.load_config import get_data_config, get_model_config, get_train_config, get_log_config
from utils.setup_logging import setup_logger
from keras import regularizers
import logging
from pathlib import Path

logger = logging.getLogger(Path(__file__).stem)

def get_referenced_embeddings(paper_id, referenced_ids, ddf):
    """Returns DataFrame containing only the embeddigns of certain IDs, sorted by publication date
            Expects ddf to come with col 'publication date' as pd.DateTimeIndex
    """
    # reindex returns NaN if index is not found in database
    referenced_works = ddf.reindex[referenced_ids].compute()
    referenced_works = referenced_works.set_index("publication_date").sort_index(kind="mergsort") # mergsort for stable sorting 
    referenced_works.filter(like = "embedding_").values
    return np.concatenate([referenced_works.reshape(1,-1), paper_id.reshape(1,-1)])


def process_partitioned_df(partitioned_df, whole_df, id_col = "id_OpenAlex", ref_works_col = "referenced_works_OpenAlex"):
    paper_ids = partitioned_df[id_col]
    referenced_works = partitioned_df[ref_works_col]

    return pd.Series([get_referenced_embeddings(x,y, whole_df) for x,y in zip(paper_ids, referenced_works)])


# def check_referenced_list_in_db(ref_list, ddf):
#     """Checks whether referenced list is in ddf index"""
#     mask = pd.isin(ref_list, ddf.index)

if __name__ == "__main__":
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu_devices))

    if gpu_devices:
        print("GPU Devices found:", gpu_devices)

    setup_logger(logger, get_log_config())
    data_config = get_data_config()
    model_config = get_model_config()
    train_config = get_train_config()

    Ninput = model_config["Ninputs"]
    Noutput = model_config["Noutputs"]

    def build_model(hp):
        l2_regularization = hp.Choice('l2_regularization', values=[0.01, 0.001, 0.0001])

        inputs = keras.Input(shape=Ninput)
        x = inputs
        for i in range(hp.Int("conv_layers", 1, 3, default=1)):
            x = keras.layers.Conv1D (
                filters = hp.Int("filters_" + str(i), 8, 64, step=8, default=16),
                kernel_size = hp.Int("kernel_size_" + str(i), 1149, 1915),
                activation = "relu",
                padding = "same",
                kernel_regularizer=regularizers.l2(l2_regularization)

            )(x)
            x = keras.layers.AveragePooling1D()(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
        
        outputs = keras.layers.Dense(Noutput, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        
        optimizer = hp.Choice("optimizer", ["adam"])

        model.compile(
            optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    


