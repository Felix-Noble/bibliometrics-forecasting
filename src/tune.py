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
import keras_tuner
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

def make_tuner(config):

    return tuner
def main():
    model = RandomForestClassifier() # temp
    setup_logger(logger, get_log_config())
    data_config = get_data_config()
    model_config = get_model_config()
    train_config = get_train_config()

    keras_model = build_model(model_config)

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

    client = Client(LocalCluster())
    logger.info(f"Dask client initialized: {client.dashboard_link}")
    logger.info("Loading database")

    data_base_loc = data_config["database_loc"] / data_config["journal"] / "parquet"
    logger.debug(f"loading from {data_base_loc}")

    id_ref_map_loc = data_config["database_loc"] / data_config["journal"] / "id_ref_map"
    ddf = dd.read_parquet(data_base_loc)
    logger.debug(ddf.index.name)
    # ddf = ddf.reset_index()
    # ddf["referenced_works_OpenAlex"] = ddf["referenced_works_OpenAlex"].apply(ast.literal_eval, meta=("referenced_works_OpenAlex", "object"))
    # id_ref_map = ddf[["id_OpenAlex", "publication_date", "referenced_works_OpenAlex"]].explode("referenced_works_OpenAlex")
    id_ref_map = dd.read_parquet(id_ref_map_loc)
    id_ref_map = id_ref_map.reset_index()
    logger.debug(len(ddf))
    ddf = ddf.dropna(subset="publication_date").persist()
    logger.debug(len(ddf))
    logger.debug("LEN AFTER DROP PUB DATE NA")
    logger.debug(len(id_ref_map))
    logger.debug(len(pd.unique(id_ref_map.index.compute())))
    
    # ddf = ddf.set_index("id_OpenAlex")
    # id_ref_map = dd.read_parquet(id_ref_map_loc)
    logger.debug(f"Database index: {ddf.index.name}")
    logger.debug(f"ID-ref map index: {id_ref_map.index.name}")
    ddf = ddf.dropna(subset = ["referenced_works_count_OpenAlex"])
    # ddf = ddf[ddf["referenced_works_count_OpenAlex"] >= 5]
    # ddf = ddf.repartition(npartitions = 64)
    logger.debug(f"Sataset (future) cols = {ddf.columns}")
    # ddf['publication_date'] = dd.to_datetime(ddf['publication_date'])

    # ddf["referenced_works_OpenAlex"] = ddf["referenced_works_OpenAlex"].apply(ast.literal_eval, meta=('referenced_works_OpenAlex', 'object'))
    logger.debug("ID-Ref Map")
    logger.debug(id_ref_map.compute())

    All_data = None    
    # ddf = ddf.sort_values("publication_date").persist()
    print(ddf.columns)
    # logger.info(f"{available_ids.shape} rows in main ddf")
    logger.info("Temporal slicing...")
    # available_ids = ddf.index.persist()
    gc.collect()
    embedding_cols = [f"embedding_{x}" for x in range(data_config["Nembeddings"])]
    # embedding_lookup_df = ddf[embedding_cols]
    
    for current_slice_end_year in range(first_slice_end_year, end_year, CV_delta):
    # for current_slice_end_year in pd.timedelta_range(first_slice_end_year, end_year):
        logger.info(f"Importing slice from temporal range: {previous_slice_end_year} {current_slice_end_year}")
        ## select range ##
        greater_than = pd.to_datetime(previous_slice_end_year, format="%Y")
        less_than = pd.to_datetime(current_slice_end_year, format="%Y")
        slice_ddf = id_ref_map[(id_ref_map["publication_date"] >= greater_than) & (id_ref_map["publication_date"] <= less_than)]
        # val_counts = slice_ddf["id_OpenAlex"].compute().value_counts()
       
        # config (move to) #
        N_REFS = 5
        N_OUT = 1
        y_cols = ["higher_than_median_year"]

        # drop ref lists less than N_REFS long
        # val_counts = val_counts[val_counts >= N_REFS]
        # logger.debug(f"VAL COUNTS {val_counts}")
        
        # Safety checks #
        if len(slice_ddf) < 1:
            logger.info("No examples found in range, skipping")
            continue

        # slice_ddf = slice_ddf.reset_index().set_index("referenced_works_OpenAlex")
        # slice_ddf = slice_ddf.reset_index().set_index("id_OpenAlex")
        # ids = [x for x in ids if x in ddf.index]
        logger.debug(slice_ddf.columns)
        # ddf["publication_date"] = dd.to_datetime(ddf["publication_date"])
        ddf_renamed = ddf.rename(columns={"publication_date":"referenced_publication_date",
                                          "referenced_works_count_OpenAlex": "ref_works_count_referenced_paper",
                                          "citation_count_OpenAlex":"referenced_paper_citation_count",
                                          })
        
        ungrouped_data = slice_ddf.merge(
            ddf_renamed[["referenced_paper_citation_count"] + embedding_cols],
            left_on="referenced_works_OpenAlex",
            right_index=True,
            how = "left"
        ).persist()
        progress(ungrouped_data)
        #ungrouped_data = ungrouped_data.set_index("id_OpenAlex").persist()
        def create_examples_X(df_partition):
   
            if df_partition.empty:
                return pd.Series(dtype='object')
            # df_partition = df_partition.reset_index()
            print(df_partition[embedding_cols])
            test = pd.DataFrame(df_partition[embedding_cols])
            print(test)
            print(df_partition[embedding_cols].to_frame())
            print(len(test))
            f = y
            # df_partition = df_partition.sort_values(by="publication_date", kind = "stable", ascending=False)
            # X_example = df_partition[embedding_cols].iloc[:N_REFS, :].values.reshape(-1,)
            # X_example = pd.Series([X_example], index=[df_partition.index[0]], name="embeddings_grouped")
            # return X_example

        def stack_and_flatten_embeddings(group_df):
            # group_df is a pandas DataFrame for a single group (e.g., all rows for id 'A')
            
            # Select only the embedding columns
            # print(group_df)
            embedding_data = group_df[embedding_cols]
            n_embeddings = embedding_data.shape[1]
            # TODO place in correct order 
            embedding_data = np.concat([embedding_data.values.flatten()[:(N_REFS * n_embeddings)], ddf.loc[group_df.name, embedding_cols].values]).flatten()
            # .values.flatten()
            # embedding_data = group_df.loc[:N_REFS, embedding_cols].values.flatten()
            # print(embedding_data)
            # # Get the underlying NumPy array and flatten it to a 1D vector
            # embedding_data = embedding_data.values.flatten()
            # print(embedding_data)
            # print(embedding_data.shape)
            return embedding_data
        
        ungrouped_data = ungrouped_data.sort_values("referenced_paper_citation_count", ascending=False)

        embeddings_grouped = ungrouped_data.groupby("id_OpenAlex").apply(stack_and_flatten_embeddings, meta=("embeddings_grouped", "object")).persist()
        embeddings_grouped = embeddings_grouped.to_frame(name="embeddings_grouped").repartition(npartitions=64).persist()
        progress(embeddings_grouped)
        logger.debug(embeddings_grouped.head())      
        logger.debug("Temporal Embedding vectors created")
        logger.debug(embeddings_grouped.compute().shape)
        
        All_slice_data = embeddings_grouped.merge(
            ddf[y_cols],
            how = "left"
        ).persist()

        logger.debug("Starting computations")
        progress(All_slice_data)
        logger.debug("Compute finished")
        All_slice_data = All_slice_data.compute()
        logger.debug(All_slice_data.shape)

        # TODO check for memory availability here, if not start batch mode. TODO write batch train mode
      
        if All_data is None:
            logger.debug("No X data found, creating X data from current slice")
            All_data = All_slice_data
        else:
            All_data = pd.concat([All_data, All_slice_data], axis=0)
        if All_data.shape[0] < 1:
            logger.info("Not enough examples, skipping")
            continue
        logger.debug(f"BEFORE DUPLICATE REMOVAL | {All_data.shape}")
        # All_data = All_data.drop_duplicates
        # All_data = All_data.loc[~All_data.index.duplicated(keep="first")]
        logger.debug(f"AFTER DUPLICATE REMOVAL | {All_data.shape}")

        test_start = pd.to_datetime(f"{current_slice_end_year}", format="%Y") - test_size
        val_start = test_start - val_size

        logger.info(f"Train test split | {start_year} | {val_start} | {test_start} | {current_slice_end_year}")

        train_ids = (All_data["publication_date"] >= start_year) & (All_data["publication_date"] < val_start)
        val_ids = (All_data["publication_date"] >= val_start) & (All_data["publication_date"] < test_start)
        test_ids = (All_data["publication_date"] >= test_start)
       
        y_train = All_data.loc[train_ids, y_cols].values
        y_val = All_data.loc[val_ids, y_cols].values
        y_test = All_data.loc[test_ids, y_cols].values
            
        X_train = All_data.loc[train_ids, "embeddings_grouped"].values
        X_val = All_data.loc[val_ids, "embeddings_grouped"].values
        X_test = All_data.loc[test_ids, "embeddings_grouped"].values

        logger.info("Data Shapes | total, train, val, test.")
        logger.info(f"All_data | {All_data.shape} ")
        logger.info(f"X train | {X_train.shape} | Y train | {y_train.shape}")
        logger.info(f"X val | {X_val.shape} | Y val| {y_val.shape}")
        logger.info(f"X test | {X_test.shape} | Y test| {y_test.shape}")
        logger.debug(All_data.head)
        logger.debug("Data sucessfully loaded")
        logger.info(All_data)
        previous_slice_end_year = current_slice_end_year
        model = RandomForestClassifier()

        peram_space = {"n_estimators": [int(x) for x in np.linspace(50,500,10)],
               "max_depth": [int(x) for x in np.linspace(5,30,3)],
               "min_samples_split": [int(x) for x in np.linspace(2,20,3)],
               "min_samples_leaf": [int(x) for x in np.linspace(1,20,3)],
               "max_features": ["sqrt", "log2"],
               'bootstrap': [False]}
        
        GS = GridSearchCV(model, peram_space, cv=5, verbose=3, n_jobs=-1, scoring="balanced_accuracy").fit(X_val, y_val)
        model.__dict__.update(GS.best_params_)
        logger.info("Fitting")
        model.fit(X_train, y_train)
        logger.info("Evaluating")
        preds = model.predict(X_test)
        BA = balanced_accuracy_score(y_test, preds)
        Prec = precision_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, preds)

        # initialise tuner # 
        tuner = keras_tuner.RandomSearch(
            hypermodel=keras_model,
            objective="val_accuracy",
            max_trials=3,
            executions_per_trial=2,
            overwrite=True,
            directory="./cache/",
            project_name=model_config["model_name"],
        )

        tuner.search_space_summary()

        # tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
        # models = tuner.get_best_models(num_models=2)
        # best_model = models[0]
        # best_model.summary()

        logger.info(f"""
          Scores:
            Accuracy: {BA}
            Precision: {Prec}
            ROC_AUC_OCO: {roc_auc}
    """)
        
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