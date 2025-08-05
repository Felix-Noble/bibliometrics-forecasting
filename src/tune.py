#src/fit.py
from src.utils.load_config import get_model_config, get_log_config, get_data_config, get_train_config
from src.utils.setup_logging import setup_logger
from pathlib import Path
import pandas as pd
import numpy as np
# import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, progress
import ast
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
def cols_generator(year, end_year, db_files, cols):
    df = duckdb.sql(f"""
    SELECT {", ".join(cols)}
    FROM read_parquet({db_file})
    WHERE publication_date_int >= {year} AND publication_date_int < {end_year}
    """).df()
    if df.shape[1] < 2:
        return df.values.reshape(-1,)
    else:
        return df.values
def group_generator(year, end_year, available_ids, db_files, sort_col, n_back, n_features=384):
    #df = pd.read_parquet(filepath, columns=["id_OpenAlex", "publication_date", "referenced_works_OpenAlex"])
    df = duckdb.sql(f"""SELECT id_OpenAlex, publication_date, referenced_works_OpenAlex 
    FROM read_parquet({db_files}) 
    WHERE publication_date_int >= {year} AND publication_date_int < {end_year}""").df()
    df["referenced_works_OpenAlex"] = df["referenced_works_OpenAlex"].apply(replace_ref_works)
    available_ids.columns = ["referenced_works_OpenAlex"] 
    df = df.explode("referenced_works_OpenAlex")
    df = df.merge(available_ids, on="referenced_works_OpenAlex", how="inner")
    df = df.groupby("id_OpenAlex")
    embedding_cols = [f"embedding_{x}" for x in range(n_features)]
    all_cols_sql = ', '.join([sort_col] + embedding_cols)
    for group_id, group_df in df: 
        
        self_embedding = duckdb.sql(f"SELECT {', '.join(embedding_cols)} FROM read_parquet({db_files}) WHERE id_OpenAlex == {group_id}").df().values.reshape(-1,)
        print(f"fetching {group_id}") 
        embeddings = duckdb.sql(f""" 
        SELECT {all_cols_sql} FROM read_parquet({db_files})
        WHERE id_OpenAlex IN ({', '.join(group_df['referenced_works_OpenAlex'].astype(str))})""").df()
        embeddings = embeddings.sort_values(sort_col, ascending=False)[embedding_cols].iloc[-n_back:, :]
        embeddings = embeddings.values.reshape(-1,)

        yield (np.concatenate([embeddings, self_embedding], dtype=np.float32),)

def create_tf_dataset(batch_size, year, step, available_ids, db_files, sort_col, n_back, n_features=384):
    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(
        lambda : group_generator(year, step, available_ids, db_files, sort_col, n_back, n_features),
        output_signature=output_signature,
    )
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=((n_back+1) * n_features,),
        padding_values=((tf.constant(0, dtype=tf.float32),))
    )

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset        


logger = logging.getLogger(Path(__file__).stem)

def process_partitioned_df():
    pass


def main():

    setup_logger(logger, get_log_config())
    data_config = get_data_config()
    model_config = get_model_config()
    train_config = get_train_config()

    n_embeddings = data_config["Nembeddings"]
    embedding_cols = [f"embedding_{x}" for x in range(n_embeddings)]
    N_REF = 5
    n_input = n_embeddings * N_REF 
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

    first_slice_end_year = start_year + train_config["start_train_size"] + val_size + test_size

    start_year = pd.to_datetime(f"{start_year}", format="%Y")
    val_size = pd.Timedelta(train_config["val_size"] * 52, "W")
    test_size = pd.Timedelta(train_config["test_size"] * 52, "W")

    previous_slice_end_year = start_year

    # keras_model = define_model(model_config)
    def build_model(hp):
        l2_regularization = hp.Choice('l2_regularization', values=[0.01, 0.001, 0.0001])

        inputs = k_layers.Input(shape=(n_input))

        x = k_layers.Reshape((N_REF, n_embeddings))(inputs)
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

    start_year_int = start_year.value
    test_size_int = test_size.value
    val_size_int = val_size.value
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
        X_test = create_tf_dataset(batch_size, test_start, test_start + test_size_int, available_ids, database_files, sort_col, N_REF, n_embeddings)
        X_val = create_tf_dataset(batch_size, val_start, test_start, available_ids, database_files, sort_col, N_REF, n_embeddings)
        X_train = create_tf_dataset(batch_size, start_year_int, val_start, available_ids, database_files, sort_col, N_REF, n_embeddings)
        test_train = X_train.as_numpy_iterator() 
        for x in test_train:
            print(x)
        continue 
        # Safety checks #
        if len(slice_id_ref) < 1:
            logger.info("No examples found in range, skipping")
            continue

        ddf_renamed = ddf.rename(columns={"publication_date":"ref_work_publication_date",
                                          "referenced_works_count_OpenAlex": "ref_work_referenced_works_count",
                                          "citation_count_OpenAlex":"ref_work_citation_count",
                                          })

        logger.info("Merging slice ddf with referenced works data")
        ungrouped_data = dd.merge(
            ddf_renamed[["ref_work_citation_count", 
                         "ref_work_referenced_works_count", 
                         "ref_work_publication_date"] + embedding_cols],
            slice_id_ref,
            left_index=True,
            # left_on="referenced_works_OpenAlex",
            right_on = "referenced_works_OpenAlex",
            # right_index=True,
            how = "inner"
        )
        # ungrouped_data = ungrouped_data.persist()
        ungrouped_data = ungrouped_data.repartition(npartitions=64).persist()
        progress(ungrouped_data)
        grouped_embedding_cols = [f"embedding_{x}" for x in range(n_embeddings * N_REFS)]
        #ungrouped_data = ungrouped_data.set_index("id_OpenAlex").persist()
        # TODO check for memory availability here, if not start batch mode. TODO write batch train mode

        if All_data is None:
            logger.debug("No X data found, creating X data from current slice")
        else:
            pass
        if All_data.shape[0] < 1:
            logger.info("Not enough examples, skipping")
            continue
    logger.info(f"Train test split | {start_year} | {val_start} | {test_start} | {current_slice_end_year}")

    # train_ids = (All_data["publication_date"] >= start_year) & (All_data["publication_date"] < val_start)
    # val_ids = (All_data["publication_date"] >= val_start) & (All_data["publication_date"] < test_start)
    # test_ids = (All_data["publication_date"] >= test_start)

    #       TODO add caching option for All data to load in pandas df in chunks, then access test train values from these views
    #       TODO add function to find index of y cols and embedding cols to improve speed here
    y_col_idx = [All_data.columns.get_loc(x) for x in y_cols]
    X_col_idx = [All_data.columns.get_loc(x) for x in [y for y in All_data.columns if y.startswith("embedding_")]]

    y_train = All_data.iloc[:val_start_idx, y_col_idx].values.ravel().astype(np.float32)
    y_val = All_data.iloc[val_start_idx:test_start_idx, y_col_idx].values.ravel().astype(np.float32)
    y_test = All_data.iloc[test_start_idx:, y_col_idx].values.ravel().astype(np.float32)

    X_train = All_data.iloc[:val_start_idx, X_col_idx].values.astype(np.float32)
    X_val = All_data.iloc[val_start_idx:test_start_idx, X_col_idx].values.astype(np.float32)
    X_test = All_data.iloc[test_start_idx:, X_col_idx].values.astype(np.float32)

    logger.info("Data Shapes | total, train, val, test.")
    logger.info(f"All_data | {All_data.shape} ")
    logger.info(f"X train | {X_train.shape} | Y train | {y_train.shape}")
    logger.info(f"X val | {X_val.shape} | Y val| {y_val.shape}")
    logger.info(f"X test | {X_test.shape} | Y test| {y_test.shape}")
    logger.debug(All_data.head)
    logger.debug("Data sucessfully loaded")
    logger.debug(All_data)
    previous_slice_end_year = current_slice_end_year


    #mlflow_logger = keras_tuner.integration.mlflow.MLflowLogger(tuner)

    tuner.search(X_train, y_train, epochs=15, validation_data=(X_val, y_val))
    models = tuner.get_best_models(num_models=2)
    best_model = models[0]
    best_model.summary()

    BA = None
    Prec = None
    roc_auc=None

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
