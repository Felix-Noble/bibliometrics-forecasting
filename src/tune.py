#src/fit.py
from os import wait
from src.utils.load_config import get_model_config, get_log_config, get_data_config, get_train_config
from src.utils.setup_logging import setup_logger
from pathlib import Path
import pandas as pd
import numpy as np
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
def replace_ref_works(string_val):
    temp = ast.literal_eval(string_val)
    return [int(x.replace("https://openalex.org/W", "")) for x in temp]

def make_id_index_map(db_loc:str, id_col:str = "id_OpenAlex"):
    database = pq.ParquetDataset(db_loc)
    id_table = database.read(columns=[id_col])
    id_index_map = {int(id_val.as_py()): i for i, id_val in enumerate(id_table[id_col])}
    return id_index_map

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
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            logger.error(f"FATAL: NaN or Inf found in features for group_id {group_id}. Skipping.")
            print("--------------------------------------------------")
            print(f"DEBUGGING INFO FOR CORRUPT DATA (group_id: {group_id})")
            print(f"Type of groupid : {type(group_id)}") 
            # 1. Show the reference IDs being looked up
            print("\nReferenced IDs in this group:")
            print(group_df)
            
            # 2. Show the exact row numbers we tried to fetch
            print("\nInteger row indices passed to pc.take():")
            print(indicies)
            
            # 3. Show the EMBEDDINGS DATAFRAME that contains the NaNs
            # This is the most important part. It will show you which rows are bad.
            print("\nEmbeddings DataFrame returned from PyArrow (contains NaNs):")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(embeddings)
            print("--------------------------------------------------")
            time.sleep(10)
            continue # Skip this sample
        logger.info(f"Collected, Elapsed = {time.time() - t1}") 
        yield (np.concatenate([embeddings], dtype=np.float32),label_map.get(group_id, -1))

def create_tf_dataset(batch_size, year, end_year, available_ids, id_index_map, db_files, sort_col, n_back, n_features=384):
    df = duckdb.sql(f"""SELECT id_OpenAlex, publication_date_int, referenced_works_OpenAlex 
    FROM read_parquet({db_files}) 
    WHERE publication_date_int >= {year} AND publication_date_int < {end_year}""").df()

    df = df.sort_values(sort_col, ascending=True)

    df["referenced_works_OpenAlex"] = df["referenced_works_OpenAlex"].apply(replace_ref_works)
    available_ids.columns = ["referenced_works_OpenAlex"] 
    df = df.explode("referenced_works_OpenAlex")
    df = df.merge(available_ids, on="referenced_works_OpenAlex", how="inner")
    df = df.groupby("id_OpenAlex")
    embedding_cols = [f"embedding_{x}" for x in range(n_features)]
    y_df = duckdb.sql(f"""
    SELECT id_OpenAlex, higher_than_median_year 
    FROM read_parquet({db_files}) 
    WHERE publication_date_int >= {year} AND publication_date_int < {end_year}
    """).df()
    label_map = pd.Series(y_df.higher_than_median_year.values, index=y_df.id_OpenAlex).to_dict()
    del y_df
 
    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(
    lambda : group_generator(Path(db_files[0]).parent, df, id_index_map, label_map, n_back, embedding_cols),
        output_signature=output_signature,
    )
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
        l2_regularization = hp.Choice('l2_regularization', values=[0.01, 0.001, 0.0001])

        inputs = k_layers.Input(shape=(n_input))

        x = k_layers.Reshape((N_REF+1, n_embeddings))(inputs)
        for i in range(hp.Int("conv_layers", 1, 2, default=1)):
            x = k_layers.Conv1D (
                filters = hp.Int("filters_" + str(i), 8, 64, step=8, default=16),
                kernel_size = hp.Int("kernel_size_" + str(i), 1149, 1915),
                activation = "relu",
                padding = "same",
                kernel_regularizer=k_reg.l2(l2_regularization)

            )(x)
            
            x = k_layers.BatchNormalization()(x)
            x = k_layers.ReLU()(x)
        x = k_layers.Flatten()(x)
        outputs = k_layers.Dense(n_output, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = hp.Choice("optimizer", ["adam"])

        model.compile(
            optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        return model


    # initialise tuner # 
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=2,
        overwrite=True,
        directory="./cache/",
        project_name=model_config["model_name"],
    )
    callback = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        mode='max'
    )

    start_year_int = start_year.value
    test_size_int = test_size.value
    val_size_int = val_size.value
    logger.info(f"{start_year_int} | {test_size_int} | {val_size_int}")
    for current_slice_end_year in range(first_slice_end_year, end_year, CV_delta):
        # for current_slice_end_year in pd.timedelta_range(first_slice_end_year, end_year):
        logger.info(f"Importing slice from temporal range: {previous_slice_end_year} {current_slice_end_year}")
        ## select range ##
        greater_than = pd.to_datetime(previous_slice_end_year, format="%Y").value
        less_than = pd.to_datetime(current_slice_end_year, format="%Y").value

        N_REFS = 5
        N_OUT = 1
        y_cols = ["higher_than_median_year"]
        sort_col = "publication_date_int"       
        batch_size = 1000


        test_start: int = pd.to_datetime(f"{current_slice_end_year}", format="%Y").value - test_size_int
        val_start: int = test_start - val_size_int

        test = create_tf_dataset(batch_size, test_start, test_start + test_size_int, available_ids, id_index_map, database_files, sort_col, N_REF, n_embeddings)
        val = create_tf_dataset(batch_size, val_start, test_start, available_ids, id_index_map, database_files, sort_col, N_REF, n_embeddings)
        train = create_tf_dataset(batch_size, start_year_int, val_start, available_ids, id_index_map, database_files, sort_col, N_REF, n_embeddings)

        logger.info("Starting search...")
        tuner.search(train, epochs=5, validation_data=val, callbacks=[callback])
        models = tuner.get_best_models(num_models=2)
        best_model = models[0]
        best_model.summary()

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
    setup_logger(logger)

    client = Client(LocalCluster())
    logger.info(f"Dask client initialized: {client.dashboard_link}")
    print(f"Dask client initialized: {client.dashboard_link}")

    logger.info("Loading database")
    # df = pd.DataFrame([])
    # df["publication_date"] = pd.DatetimeIndex(range(0, 20))
    # df["referenced_works_OpenAlex"] = pd.Series([[{i+1}] * i for i in range(20)])
    # df["embedding"] = pd.Series([i] for i in range(20))
    # df["idi"] = df.index
    # df["id"] = df.index
    # ddf = dd.from_pandas(df)

    ddf = dd.read_csv("./data/test_data.csv")
    print(len(ddf))
    ddf = ddf[ddf["referenced_works_count"] > 0]
    print(len(ddf))

    ddf["referenced_works_OpenAlex"] = ddf["referenced_works_OpenAlex"].apply(ast.literal_eval, meta=('referenced_works_OpenAlex', 'object'))
    print(f"Referenced works lens: {ddf['referenced_works_OpenAlex'].apply(len, meta=('referenced_works_count_OpenAlex', 'int64')).compute()}")

    def format_id_OpenAlex(x):
        return int(x.split("W")[-1])

    def format_list_id_OpenAlex(y):
        return [format_id_OpenAlex(x) for x in y]

    ddf["id_OpenAlex"] = ddf["id_OpenAlex"].apply(format_id_OpenAlex, meta=("id_OpenAlex", "int"))
    ddf["referenced_works_OpenAlex"] = ddf["referenced_works_OpenAlex"].apply(format_list_id_OpenAlex, meta=("referenced_works_OpenAlex", "object"))

    expl_ddf = ddf[["id_OpenAlex", "referenced_works_OpenAlex"]]

    ddf = ddf.set_index("id_OpenAlex")

    expl_ddf = ddf[["publication_date", "referenced_works_OpenAlex"]].explode("referenced_works_OpenAlex").persist()

    print("Exploded")
    print(expl_ddf.index.name)

    # expl_ddf = expl_ddf.set_index("id_OpenAlex").persist()
    # print(expl_ddf.head())

    def check_isin_or_NA(id: pd.Series) -> pd.Series:
        # if isinstance(series, str):
        #     check = pd.Series(series)

        if not np.isin(id, ddf.index):
            return pd.NA
        return id

    def check_is_in_short(index_val):
        if np.isin(expl_ddf[index_val], ddf.index):
            return pd.NA

    expl_ddf = expl_ddf.merge(
        ddf[["title"]+ [col for col in ddf.columns if col.startswith("embed")]],
        left_index=True,
        right_index=True,
        how='left'
    )

    # expl_ddf["publication_date"] = ddf["publication_date"][expl_ddf.index]
    # expl_ddf["title_referencing_paper"] = ddf.loc[expl_ddf.index, "title"]

    # expl_ddf["title_referencing_paper"] = ddf["title"][expl_ddf.index]

    expl_ddf["referenced_works_OpenAlex"] = expl_ddf["referenced_works_OpenAlex"].apply(check_isin_or_NA, meta=("referenced_works_OpenAlex", "string"))

    # expl_ddf["referenced_works_is_in_index"] = expl_ddf["referenced_works_OpenAlex"].apply(, meta=("referenced_works_OpenAlex", "string"))

    expl_ddf = expl_ddf.dropna(subset=["referenced_works_OpenAlex"]).persist()

    print(expl_ddf.compute())

    expl_ddf = expl_ddf.reset_index().persist()#.set_index("referenced_works_OpenAlex")

    # test if multi column indexing works 

    ddf["embedding1"] = 1
    ddf["embedding2"] = 2

    ddf_renamed = ddf.rename(columns={
        "title": "title_paper_referenced",
        "doi": "doi_paper_referenced"
    })

    # 2. Merge 'expl_ddf' with the selected columns from the renamed 'ddf'.
    # The merge happens on the index of both DataFrames.
    # expl_ddf = expl_ddf.merge(
    # ddf_renamed[["title_paper_referenced", "doi_paper_referenced"]],
    # left_index=True,
    # right_index=True,
    # how='left'  # 'left' ensures you keep all original rows from expl_ddf
    # )

    expl_ddf = expl_ddf.merge(
        ddf_renamed[["title_paper_referenced", "doi_paper_referenced"]],
        left_index=["id_OpenAlex"],
        right_index=["referenced_works_OpenAlex"],
        how='left'  # 'left' ensures you keep all original rows from expl_ddf
    )

    # expl_ddf[["title_paper_referenced", "doi_paper_referenced"]] = ddf.loc[expl_ddf.index,["title", "doi"]]

    print(len(expl_ddf))
    expl_ddf = expl_ddf.sort_values("publication_date")
    print(len(expl_ddf))
    print(expl_ddf.compute())

    print(expl_ddf.index.name, expl_ddf.index)

    expl_ddf.reset_index().compute().to_csv("./data/test_id-ref-map.csv")

    # rows = expl_ddf[expl_ddf['id'].isin([2])]["referenced_works_OpenAlex"].compute()

    # rows = expl_ddf.loc[[2]]["referenced_works_OpenAlex"].compute()

    # print(rows)

    # print(ddf[ddf["id"].isin(rows.values)].compute())

    # ddf = dd.read_parquet(data_config["database_loc"])
    logger.debug(f"Database index: {ddf.index.name}")
    # ddf = ddf.repartition(npartitions = 64)
    logger.debug(f"Sataset (future) cols = {ddf.columns}")
    ddf['publication_date'] = dd.to_datetime(ddf['publication_date'])



    # main()
