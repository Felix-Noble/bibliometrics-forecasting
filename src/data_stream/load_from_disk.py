from os import replace, wait
import pyarrow.compute as pc
import pyarrow.parquet as pq 
import pyarrow.dataset as ds
import pyarrow as pa
import duckdb
from pathlib import Path
import psutil
import numpy as np 
import pandas as pd
import ast
import gc
def replace_ref_works(string_val):
    temp = ast.literal_eval(string_val)
    return [int(x.replace("https://openalex.org/W", "")) for x in temp]

def make_id_index_map(db_loc:str, id_col:str = "id_OpenAlex"):
    database = pq.ParquetDataset(db_loc)
    id_table = database.read(columns=[id_col])
    id_index_map = {int(id_val.as_py()): i for i, id_val in enumerate(id_table[id_col])}
    return id_index_map

def load_cols_all(
    files: list,
    cols:list | str = ["id_OpenAlex"],
):

    if len(cols) == 1:
        cols = cols[0]
    else:
        cols = ", ".join(cols)

    data = duckdb.sql(f"""SELECT {cols} FROM read_parquet({files})
                        """).df()
    return data

def load_cols_within_range(
    t_start: int,
    t_end: int, 
    files: list,
    cols:list | str = ["id_OpenAlex"],

):
    if len(cols) == 1:
        cols = cols[0]
    else:
        cols = ", ".join(cols)

    data = duckdb.sql(f"""SELECT {cols} FROM read_parquet({files})
                            WHERE publication_date_int >= {t_start} AND publication_date_int < {t_end} 
                        """).df()
    return data

def generate_examples_in_range(
    dir: str,
    t_start: int,
    t_end: int,
    y_cols: list,
    n_features: int,
    max_timepoints: int,
    id_index_map: dict,
    pad: str = "previous",
    pad_value: float = 0.0,
    sort_col: str = "publication_date_int",
    id_col: str = "id_OpenAlex",
    features_dtype = np.float32(),
    VRAM_MAX: int = 4500 * (1024 **2)
):
    # generate file names 
    files = [str(x) for x in Path(dir).glob("*.parquet")]
    feature_cols = [f"embedding_{x}" for x in range(n_features)]
    # DB and column lookup tables
    DB = pq.read_table(dir, memory_map=True)
    feature_lookup_table = DB.select([id_col, "publication_date_int"] + feature_cols + y_cols) 
    ref_works_lookup_table = DB.select([id_col, "referenced_works_OpenAlex", "publication_date_int"])

    available_ids = load_cols_all(files, [id_col, "publication_date_int"])
    id_mask = (available_ids["publication_date_int"] >= t_start) & (available_ids["publication_date_int"] < t_end)
    ids_in_range = available_ids.loc[id_mask, :]

    # logger.info(f"{ids_in_range.shape[0]} examples in split")
    
    available_ids.rename(columns= {id_col: "referenced_works_OpenAlex"}, inplace=True)
    
    id_idxs = [id_index_map.get(id) for id in ids_in_range[id_col]] 
    referenced_works = pc.take(ref_works_lookup_table, pa.array(id_idxs, type=pa.int64())).to_pandas()
    if referenced_works.index.name == id_col:
        referenced_works = referenced_works.reset_index()
    # generate referenced works 
    referenced_works["referenced_works_OpenAlex"] = referenced_works["referenced_works_OpenAlex"].apply(replace_ref_works)
    referenced_works = referenced_works.sort_values(sort_col, ascending=True)
    referenced_works = referenced_works.explode("referenced_works_OpenAlex")
    referenced_works = referenced_works.merge(available_ids, on = "referenced_works_OpenAlex", how="inner")
    referenced_works['ref_rank'] = referenced_works.groupby(id_col).cumcount()
    referenced_works = referenced_works[referenced_works["ref_rank"] < max_timepoints]

    total_examples = sum(referenced_works["ref_rank"] == 0)
    examples_finished = 0 
    batch = 0
    example_size = (max_timepoints+1) * n_features * features_dtype.nbytes * 16
    full_cols = pd.MultiIndex.from_product([feature_cols, range(max_timepoints)], names=[None, "ref_rank"])
    
    while examples_finished < total_examples:
        # determine batch size 
        mem = psutil.virtual_memory()
        available = min(mem.available, VRAM_MAX)
        batch_size = available // example_size

        # add referenced works features 
        batch_df = referenced_works.iloc[examples_finished : examples_finished + batch_size, :].copy()
        import time
        feature_idxs = [id_index_map.get(id) for id in batch_df["referenced_works_OpenAlex"]] 
        batch_features = pc.take(feature_lookup_table, pa.array(feature_idxs, type=pa.int64())).to_pandas()
        if batch_features.index.name == id_col:
            batch_features = batch_features.reset_index()

        batch_features.rename(columns = {id_col: "referenced_works_OpenAlex"}, inplace = True)

        batch_df = batch_df.merge(batch_features, on = "referenced_works_OpenAlex", how="inner")
        del batch_features

        batch_df = batch_df.pivot_table(
                            index = id_col,
                            columns = "ref_rank",
                            values = feature_cols,
        )
        # Extend columns if less than expected
        if batch_df.shape[1] < max_timepoints * n_features:
            batch_df = batch_df.reindex(columns=full_cols)
        # Reorder
        batch_df.columns = [f'{col[0]}_{col[1]}' for col in batch_df.columns]
        batch_df = batch_df[[f'embedding_{x}_{y}' for y in range(max_timepoints-1, -1, -1) for x in range(n_features)]]
        # Pad 
        batch_df = batch_df.fillna(pad_value)

        # Add self features
        feature_idxs = [id_index_map.get(id) for id in batch_df.index] 
        self_features = pc.take(feature_lookup_table, pa.array(feature_idxs, type=pa.int64())).to_pandas()
        batch_df = batch_df.merge(self_features, left_index = True, right_on = id_col, how = 'inner', suffixes = ("_ref", "_source"))
        batch_df.rename(columns = {k : v for k,v in zip(y_cols, ["label_" + x for x in y_cols])}, inplace=True)
        del self_features
        #print(f"total_examples = {total_examples} referenced_works shape= {referenced_works.shape}, BATCH: {batch}, examps finished {examples_finished}, batch_size {batch_size}, availalbe_mem{available}, batch_df_shape {batch_df.shape}")
        gc.collect() 
        yield batch_df, batch
        examples_finished += batch_df.shape[0]
        batch += 1
 
if __name__ == "__main__":
    db_loc = "./data/ACS"
    id_index_map = make_id_index_map(db_loc)
    generate_examples_in_range(t_start=0, 
                               t_end=int(10e6), 
                               y_cols=["higher_than_median_year"], 
                               dir=db_loc, 
                                n_features=384, 
                               max_timepoints=50)



