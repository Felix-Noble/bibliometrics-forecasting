#src/fit.py
# from load_data import get_sorted_id_ref_map
# from model_builder import build_model
from utils.load_config import get_model_config, get_log_config, get_data_config, get_train_config
from utils.setup_logging import setup_logger
from pathlib import Path
import pandas as pd
import numpy as np
# import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import ast
import logging

import gc

logger = logging.getLogger(Path(__file__).stem)

def process_partitioned_df():
    pass

def main():
    setup_logger(logger, get_log_config())
    data_config = get_data_config()
    model_config = get_model_config()
    train_config = get_train_config()
    model = build_model(model_config)

    start_year = train_config["start_year"]

    val_size = train_config["val_size"]
    test_size = train_config["test_size"]
    end_year = train_config["end_year"]
    CV_delta = train_config["CV_delta"]

    first_slice_end_year = start_year + train_config["start_train_size"] + val_size + test_size

    previous_slice_end_year = start_year

    client = Client(LocalCluster())
    logger.info(f"Dask client initialized: {client.dashboard_link}")
    logger.info("Loading database")
    
    ddf = dd.read_parquet(data_config["database_loc"])
    logger.debug(f"Database index: {ddf.index.name}")
    ddf = ddf.dropna(subset = ["referenced_works_count_OpenAlex"])
    # ddf = ddf.repartition(npartitions = 64)
    logger.debug(f"Sataset (future) cols = {ddf.columns}")
    ddf['publication_date'] = dd.to_datetime(ddf['publication_date'])

    ddf["referenced_works_OpenAlex"] = ddf["referenced_works_OpenAlex"].apply(ast.literal_eval, meta=('referenced_works_OpenAlex', 'object'))
    #ddf = ddf.persist()

    # index = "id_OpenAlex" # TODO move to config 
    # try:
    #     cache_file = False
    #     id_ref_map = dd.read_parquet(
    #         Path(__file__).parent / ".cache" / "id_ref_map"
    #     )
    #     logger.info("Id Ref Map Loaded from cache")
    # except FileNotFoundError as e:
    #     logger.info(f"Creating id ref map | no file found : {e}")
    #     cache_file = True
    # except Exception as e:
    #     logger.error(f"Creating id ref map | Error: {e}")
    #     cache_file = True
   
    #     # logger.info(f"Map creation complete. Caching to {Path('./.cache/id_ref_map')}")
    
    # logger.debug(f'{Path(__file__).parent / "cache" / "id_ref_map"}')
    
    # if cache_file:
    #     id_ref_map = ddf.loc[:, [index, 'referenced_works_OpenAlex']].explode('referenced_works_OpenAlex')
    #     id_ref_map = id_ref_map.set_index(index).persist()

    #     id_date_map = ddf.loc[:, [index, 'publication_date']]

    #     def map_id_date(id):
    #         return id_date_map[id]
        
    #     id_ref_map["publication_date"] = pd.Series(id_ref_map.index).apply(map_id_date)
        
    #     # id_ref_map.loc[id_date_map[index], "publication_date"] = id_date_map["publication_date"] 
        
    #     print(id_ref_map.compute())

    #     """indexed_publication_date = ddf[index, "publication_date"]
    #     id_ref_map = dd.merge(
    #         left=id_ref_map,
    #         right=indexed_publication_date,
    #         left_on='id_OpenAlex',
    #         right_index=True,
    #         how='inner' # Use 'inner' to drop references that weren't found
    #     ).persist()
    #     del indexed_publication_date

    #     id_ref_map["publication_date"] = dd.to_datetime(id_ref_map["publication_date"])

    #     id_ref_map = id_ref_map.sort_values(['publication_date','id_OpenAlex']).persist()
    #     id_ref_map = id_ref_map.set_index("publication_date").persist()"""

    #     # id_ref_map = get_sorted_id_ref_map(ddf[[index, "referenced_works_OpenAlex"]], ddf)
        
    #     #id_ref_map = id_ref_map.persist()
    #     logger.debug("CHECK")
    #     id_ref_map.to_parquet(
    #         Path(__file__).parent / "cache" / "id_ref_map",
    #         engine='pyarrow',
    #         write_index=True,
    #         overwrite=True,
    #     )
    #     logger.info("Map cached.")

    
    # logger.info(f"Map loaded into memory ~ {len(id_ref_map)} items")

    X_data = pd.DataFrame([])
    y_data = pd.DataFrame([])
    ddf = ddf.set_index("id_OpenAlex")
    # ddf = ddf.sort_values("publication_date").persist()

    # logger.info(f"{available_ids.shape} rows in main ddf")
    logger.info("startuing slice")
    # available_ids = ddf.index.persist()
    gc.collect()
    embedding_cols = [f"embedding_{x}" for x in range(data_config["Nembeddings"])]
    embedding_lookup_df = ddf[embedding_cols]
    
    for current_slice_end_year in range(first_slice_end_year, end_year, CV_delta):
        logger.info(f"{previous_slice_end_year, current_slice_end_year}")
        greater_than = pd.to_datetime(previous_slice_end_year, format="%Y")
        less_than = pd.to_datetime(current_slice_end_year, format="%Y")
        slice_ddf = ddf[(ddf["publication_date"] > greater_than) & (ddf["publication_date"] <= less_than)]
        # slice_ddf = slice_ddf.reset_index().set_index("id_OpenAlex")
        # ids = [x for x in ids if x in ddf.index]
        sliced_ids = slice_ddf.reset_index()["id_OpenAlex"].compute()

        # print(sliced_ids)
        refs = slice_ddf[["publication_date", "referenced_works_OpenAlex"]]

        refs_expl = refs.explode("referenced_works_OpenAlex")

        merged_citations = dd.merge(
                left=refs_expl,
                right=embedding_lookup_df,
                left_on='referenced_works_OpenAlex', # The key from the exploded references
                right_index=True,       # The key from our lookup table ('id_OpenAlex')
                how='inner' # Use 'inner' to only keep found references
            )
            
        def stack_embeddings_cols(grouped_ddf_entries):
            temp = grouped_ddf_entries.sort_values("publication_date")
            temp = temp[embedding_cols].values.tolist()
            return temp #+ slice_ddf[grouped_ddf_entries.index, embedding_cols].values.tolist()
        
        logger.debug("Starting groupby operation")
        grouped_citations = merged_citations.groupby(merged_citations.index).apply(stack_embeddings_cols, meta=("embeddings_list", "object"))
        # slice_X_data = merged_citations.groupby(merged_citations.index)[embedding_cols].compute().values.tolist()
        slice_X_data = grouped_citations.compute()
        # Get the corresponding y_data (citation counts of the original papers in the slice)
        slice_y_data = slice_ddf['citation_count_OpenAlex'].compute()
        
        print(slice_X_data)
        sliced_ids = slice_X_data.index.to_series().compute()
        print(sliced_ids)
        print(merged_citations.index.to_series())
        print("INDEX COMPARISON ABOVE")
        for i, x in enumerate(slice_X_data):
            print(len(x))
            id = sliced_ids.iloc[i]
            print(id)
            print(ddf.loc[id, "referenced_works_count_OpenAlex"].compute())
            print("::")

        # logger.debug(less_than)
        # new_ids.extend(sliced_vals["id_OpenAlex"])

        """
        ddf_slice = ddf.loc[new_ids]
        new_data = ddf_slice.map_partitions(
            process_partitioned_df,
            meta = pd.DataFrame(columns=[f"embedding_{x}" for x in range(data_config["Nembeddings"])], dtype=np.float32)
        )

        # append only new data to X and Y
        new_X_data = ddf_slice["higher_than_median_year"].compute()
        new_y_data = ddf_slice["higher_than_median_year"].compute()
        
        X_data = pd.concat([X_data, new_X_data], axis = 0)
        y_data = pd.concat([y_data, new_y_data], axis = 0)

        test_start = end_year - test_size
        val_start = test_size - val_size

        X_test = X_data.loc[test_start : end_year]
        X_val = X_data.loc[val_start : test_start]
        X_test = X_data.loc[start_year : val_start]

        y_test = y_data.loc[test_start : end_year]
        y_val = y_data.loc[val_start : test_start]
        y_test = y_data.loc[start_year : val_start]

        logger.info(f"{X_data.shape}")
        logger.info(X_data.head)
"""
        previous_slice_end_year = current_slice_end_year

        # tuner = keras_tuner.Hyperband(
        # hypermodel = build_model,
        # objective = "val_accuracy",
        # max_epochs = 50,
        # factor = 3,
        # distribution_strategy=tf.distribute.MirroredStrategy(),
        # directory=f"results_dir{end_year}",
        # project_name="mnist",
        # overwrite=True,
        # seed=2025 )


"""

    id_year_df = ddf[['id_OpenAlex', 'publication_year']]
    logger.info("Creating the ID-to-Year lookup map...")
    id_to_year_map = id_year_df.compute().set_index('id_OpenAlex')['publication_year'].to_dict()
    del id_year_df

    year_id_map = {year:[] for year in range(start_year, end_year + 1)}
    for id,year in id_to_year_map.items():
        if year >= start_year and year <= end_year:
            year_id_map[year].append(id)

    """
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
    
    def foo(x):
        return int(x.split("W")[-1])
    
    def foo2(y):
        return [foo(x) for x in y]
    
    ddf["id_OpenAlex"] = ddf["id_OpenAlex"].apply(foo, meta=("id_OpenAlex", "int"))
    ddf["referenced_works_OpenAlex"] = ddf["referenced_works_OpenAlex"].apply(foo2, meta=("referenced_works_OpenAlex", "object"))

    expl_ddf = ddf[["id_OpenAlex", "referenced_works_OpenAlex"]]
    
    
    
    ddf = ddf.set_index("id_OpenAlex")

    ddf = ddf.set_index("id_OpenAlex")



    ddf = ddf.set_index("id_OpenAlex")

    expl_ddf = ddf[["publication_date", "referenced_works_OpenAlex"]].explode("referenced_works_OpenAlex").persist()

    ddf = ddf.set_index("id_OpenAlex")

    expl_ddf = ddf[["publication_date", "referenced_works_OpenAlex"]].explode("referenced_works_OpenAlex").persist()

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
    expl_ddf = expl_ddf.merge(
    ddf_renamed[["title_paper_referenced", "doi_paper_referenced"]],
    expl_ddf = expl_ddf.merge(
    expl_ddf = expl_ddf.merge(
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