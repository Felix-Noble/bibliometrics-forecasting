#src/fit.py
from src.model_builder import build_model
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

import mlflow
import keras
import keras_tuner
import keras.layers as k_layers
import keras.regularizers as k_reg
import tensorflow as tf
import warnings 
import gc
# temp #
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
shuffle_logger = logging.getLogger("distributed.shuffle._scheduler_plugin")
# Set its level to ERROR to suppress warnings
shuffle_logger.setLevel(logging.ERROR)
# dask.config.set({'logging': {'distributed': logging.ERROR}})

logger = logging.getLogger(Path(__file__).stem)

def process_partitioned_df():
    pass

def create_id_ref_map(ddf):
    
    id_ref_map = ddf[["publication_date", "referenced_works_OpenAlex"]].explode("referenced_works_OpenAlex")
    
    id_ref_map = id_ref_map.reset_index()
    id_ref_map = id_ref_map.dropna()
    id_ref_map= id_ref_map.repartition(npartitions=64).persist()
    progress(id_ref_map)
    logger.info(f"ID-Ref map created | nrows = {len(id_ref_map)}")
    # id_ref_map.to_parquet(
    #     "./cache/exploded_ddf",
    #     write_index=True,
    #     overwrite=True,
    #     schema={"id_OpenAlex": "large_string",
    #             "publication_date":"timestamp[ns]",
    #             "referenced_works_OpenAlex": "string",
    #             "__null_dask_index__": "int64"}
    # ) 
    logger.info("Checking id membership")
    
    # TODO add save/load functions and universalise schema
    # filter out id's that aren't in the main ddf index 
    # id_ref_map = dd.read_parquet("./cache/exploded_ddf",
    #                              schema={"id_OpenAlex": "large_string",
    #                                     "publication_date":"timestamp[ns]",
    #                                     "referenced_works_OpenAlex": "string",
    #                                     "__null_dask_index__": "int64"})
    id_ref_map = id_ref_map.merge(
        ddf[[]], # Creates an empty DataFrame with just the index of ddf
        left_on="referenced_works_OpenAlex",
        right_index=True,
        how="inner",
        
    ).persist()
    progress(id_ref_map)
    logger.info(f"Referenced papers not in database dropped | nrows = {len(id_ref_map)}")
    # TODO build tool for commented before scaling 
    # expl_ddf["publication_date"] = dd.to_datetime(expl_ddf["publication_date"])
    # expl_ddf = expl_ddf.sort_values("publication_date").persist()
    id_ref_map["referenced_works_OpenAlex1"] = id_ref_map["referenced_works_OpenAlex"].apply(lambda x : np.int64(x.replace("https://openalex.org/W", "")), 
                                                                                                meta=("referenced_works_OpenAlex1","int64")).persist()
    
    logger.info("Saving id ref map")
    id_ref_map = id_ref_map.repartition(npartitions=64).persist()
    progress(id_ref_map)
    id_ref_map.to_parquet(
        "./cache/" / "id_ref_map",
        write_index = True,
        overwrite = True,
        schema={"id_OpenAlex": "int64",
                "publication_date":"timestamp[ns]",
                "referenced_works_OpenAlex": "int64",
                "__null_dask_index__": "int64"}
    )

    logger.info("ID-ref map saved")

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
    mlflow.set_experiment("Test Run") # TODO add each of hardcoded to config 
    tf.random.set_seed(2025)

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

    client = Client(LocalCluster())
    logger.info(f"Dask client initialized: {client.dashboard_link}")

    logger.info("Loading database")
    data_base_loc = data_config["database_loc"] / data_config["journal"] / "parquet"
    logger.debug(f"loading from {data_base_loc}")

    ddf = dd.read_parquet(data_base_loc,
                          columns=["publication_date", "citation_count_OpenAlex", "referenced_works_count_OpenAlex", "higher_than_median_year"] + embedding_cols,
                          dtypes={"id_OpenAlex": "int64"})
    
    id_ref_map_loc = data_config["database_loc"] / data_config["journal"] / "id_ref_map"
    id_ref_map = dd.read_parquet(id_ref_map_loc,
                                 dtypes={"id_OpenAlex": "int64",
                                         "referenced_works_OpenAlex": "int64"})

    if logger.level == logging.DEBUG:
       

        ddf.reset_index()[["id_OpenAlex"] + embedding_cols[:5]].compute().to_csv("./cache/ddf_index_test.csv")

    # data safety checks #
    pass

    # data safety fixes #
     
    # ddf["referenced_works_OpenAlex"] = ddf["referenced_works_OpenAlex"].apply(ast.literal_eval, meta=('referenced_works_OpenAlex', 'object'))
    logger.debug("ID-Ref Map")
    logger.debug(id_ref_map.compute())

    All_data = None    
   
    logger.info("Temporal slicing...")
    gc.collect()
    for current_slice_end_year in range(first_slice_end_year, end_year, CV_delta):
    # for current_slice_end_year in pd.timedelta_range(first_slice_end_year, end_year):
        logger.info(f"Importing slice from temporal range: {previous_slice_end_year} {current_slice_end_year}")
        ## select range ##
        greater_than = pd.to_datetime(previous_slice_end_year, format="%Y")
        less_than = pd.to_datetime(current_slice_end_year, format="%Y")
        slice_id_ref = id_ref_map[(id_ref_map["publication_date"] >= greater_than) & (id_ref_map["publication_date"] <= less_than)]

        if logger.level == logging.DEBUG:
            slice_id_ref.compute().to_csv("./cache/slice_ddf.csv")
        # config (move to) #
        N_REFS = 5
        N_OUT = 1
        y_cols = ["higher_than_median_year"]
       
        # id_counts = slice_id_ref.groupby("id_OpenAlex")["id_OpenAlex"].transform('count', meta=pd.Series(dtype='int64', name='id_counts')).persist()
        id_counts = slice_id_ref.groupby("id_OpenAlex")["id_OpenAlex"].count()
        id_counts = id_counts[id_counts >= N_REFS].persist()
        id_counts = id_counts.to_frame(name="count")
        
        slice_id_ref = dd.merge(slice_id_ref, id_counts, on="id_OpenAlex", how="inner").persist()
        # val_counts = slice_id_ref["id_OpenAlex"].compute().value_counts()
 
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
        
        if logger.level == logging.DEBUG:
            # Inside your loop, after slice_ddf is created and its index is set

            logger.debug(f"slice_ddf index name: {slice_id_ref.index.name}")
            logger.debug(f"ddf index name: {ddf.index.name}")

            # Compute unique indices to compare
            slice_refs_in_range = slice_id_ref["referenced_works_OpenAlex"].unique().compute()
            main_ddf_ids = ddf.index.compute()
            logger.debug(f"Number of each id | slice refs {len(slice_refs_in_range)} | main ddf idx {len(main_ddf_ids)} ")
                       
            slice_refs_in_range.to_csv("./cache/slice_refs_in_range.csv")
            # ddf.compute().to_csv("./cache/ddf2.csv")
           
            # Check how many referenced IDs are actually present in the main ddf
            num_found = slice_refs_in_range.isin(main_ddf_ids).sum()
            total_refs_in_range = len(slice_refs_in_range)

            logger.info(f"Overlap check: {num_found} out of {total_refs_in_range} 'referenced_works_OpenAlex' IDs in slice_ddf's index are found in ddf's 'id_OpenAlex' index.")

            if num_found == 0:
                logger.warning("No overlap detected! This is why your embeddings are NaN.")
            elif num_found < total_refs_in_range:
                logger.warning(f"Partial overlap: {total_refs_in_range - num_found} 'referenced_works_OpenAlex' IDs are NOT found in ddf's index.")
            
            # ddf_renamed.compute().to_csv("./cache/ddf_renamed.csv")
            ungrouped_data.compute().to_csv("./cache/ungrouped_data.csv")

            if len(ungrouped_data) < 1 :
                logger.error(ungrouped_data.compute())
                client.close()
                raise ValueError("no data matched in ungrouped data")

        grouped_embedding_cols = [f"embedding_{x}" for x in range(n_embeddings * N_REFS)]
        #ungrouped_data = ungrouped_data.set_index("id_OpenAlex").persist()
        def stack_and_flatten_embeddings(group_df):
            # group_df is a pandas DataFrame for a single group (e.g., all rows for id 'A')
            
            embedding_data = group_df[embedding_cols]
            
            embedding_out = np.concatenate([[group_df.name], embedding_data.iloc[-N_REFS:, :].values.reshape(-1,)]).reshape(1,-1)
            df = pd.DataFrame(embedding_out, columns = ["ID"] + grouped_embedding_cols)
            return df
        ungrouped_data = ungrouped_data.sort_values("ref_work_citation_count", ascending=False)
        embeddings_grouped = ungrouped_data.groupby("id_OpenAlex").apply(stack_and_flatten_embeddings, meta=pd.DataFrame([], columns = ["ID"] + grouped_embedding_cols, dtype=np.float32))
        # TODO move index assignment to dask groupby.apply operation
       
        embeddings_grouped = embeddings_grouped.reset_index().persist().drop(columns = "index").persist()
        
        embeddings_grouped = embeddings_grouped.set_index("ID").persist()

        logger.info("Creating Temporal Embedding vectors")
        embeddings_grouped = embeddings_grouped.repartition(npartitions=64).persist()
        progress(embeddings_grouped)
        # logger.debug(embeddings_grouped.compute().iloc[:5, :])      
        if logger.level == logging.DEBUG:
            embeddings_grouped.compute().to_csv("./test/embeddings_grouped_test.csv")
            
        # embeddings_grouped = embeddings_grouped.set_index("id_OpenAlex")
        logger.info("Merging embeddings with y data")
        All_slice_data = embeddings_grouped.merge(
            ddf[y_cols + ["publication_date"]],
            left_index=True,
            right_index=True,
            how = "inner"

        ).persist()
        # All_slice_data = All_slice_data.repartition(npartitions=64)
        progress(All_slice_data)

        All_slice_data = All_slice_data.compute()

        # TODO check for memory availability here, if not start batch mode. TODO write batch train mode

        if All_data is None:
            logger.debug("No X data found, creating X data from current slice")
            All_data = All_slice_data
        else:
            All_data = pd.concat([All_data, All_slice_data], axis=0)
        if All_data.shape[0] < 1:
            logger.info("Not enough examples, skipping")
            continue

        for col in embedding_cols:
            # 'coerce' will turn any value that cannot be converted to a number into NaN
            All_data[col] = pd.to_numeric(All_data[col], errors='coerce')

        # Do the same for your target variable if it could contain non-numeric values
        for col in y_cols:
            All_data[col] = pd.to_numeric(All_data[col], errors='coerce')
            
        All_data = All_data.dropna()
        All_data = All_data.sort_values("publication_date")
        if logger.level == logging.DEBUG:
            All_data.to_csv("./test/All_data.csv")
        all_dates = All_data["publication_date"]
        
        test_start = pd.to_datetime(f"{current_slice_end_year}", format="%Y") - test_size
        val_start = test_start - val_size

        val_start_idx = all_dates.searchsorted(val_start)
        test_start_idx = all_dates.searchsorted(test_start)

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
