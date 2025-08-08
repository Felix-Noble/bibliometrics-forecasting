#/src/load_data.py
from src.utils.load_config import get_log_config
from src.utils.setup_logging import setup_logger
import pyarrow.compute as pc
import pyarrow.parquet as pq 
import pyarrow as pa
import tensorflow as tf 
import pandas as pd 
import numpy as np
import time
import duckdb
import ast
import os
import logging
from pathlib import Path

logger = logging.getLogger(Path(__file__).stem)

def replace_ref_works(string_val):
    temp = ast.literal_eval(string_val)
    return [int(x.replace("https://openalex.org/W", "")) for x in temp]

def make_id_index_map(db_loc:str, id_col:str = "id_OpenAlex"):
    database = pq.ParquetDataset(db_loc)
    id_table = database.read(columns=[id_col])
    id_index_map = {int(id_val.as_py()): i for i, id_val in enumerate(id_table[id_col])}
    return id_index_map

def group_generator_pivot(db_dir: Path, main_df, id_index_map:dict, label_map:dict, n_back: int, feature_cols:list, batch_size:int =1024):
# TODO optimise memory allocations in this section (overwrite existing variables during transformations)
    main_db = pq.read_table(db_dir, memory_map=True)
    col_lookup_table = main_db.select(["id_OpenAlex"] + feature_cols) 
    all_ids = main_df['id_OpenAlex'].unique()
    logger.info(f"{all_ids.shape[0]} examples in split")
    for i in range(0, all_ids.shape[0], batch_size):
        batch_ids = all_ids[i:i + batch_size]
        df = main_df[main_df['id_OpenAlex'].isin(batch_ids)].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # 5. Rank references within each group and keep the top N
        df['ref_rank'] = df.groupby('id_OpenAlex').cumcount()
        df = df[df['ref_rank'] < n_back]
        ref_indicies = [id_index_map.get(id) for id in df["referenced_works_OpenAlex"]] 
        embeddings_ref = pc.take(col_lookup_table, pa.array(ref_indicies, type=pa.int64())).to_pandas()
        
        df = df.merge(embeddings_ref, on="id_OpenAlex", how="inner")
        
        #del embeddings_ref, ref_indicies
        
        features_df = df.pivot_table(
                index='id_OpenAlex',
                columns='ref_rank',
                values=feature_cols
            ).dropna()
            # Flatten the multi-level column index
        features_df.columns = [f'{col[0]}_{col[1]}' for col in features_df.columns]
        
        features_df = features_df[[f"embedding_{x}_{y}" for y in range(n_back) for x in range(len(feature_cols))]]
        self_indicies = [id_index_map.get(id) for id in features_df.index]
        self_embeddings = pc.take(col_lookup_table, pa.array(self_indicies, type=pa.int64())).to_pandas()
        
        # 7. Add embeddings for the source paper itself
        final_df = features_df.merge(
            self_embeddings,
            left_on="id_OpenAlex",
            right_index=True,
            how='inner',
            suffixes=('_ref', '_source')
        )
        yield (final_df.values, np.array([label_map.get(id) for id in final_df.index]))
    
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
       # if np.isnan(embeddings).any() or np.isinf(embeddings).any():
       #     raise ValueError("Embedddings contain NaN or Inf") 
        yield (np.concatenate([embeddings], dtype=np.float32), label_map.get(group_id, -1))

def create_tf_dataset(db_dir, year, end_year, sort_col, n_back, n_features=384, cache_dir = "./tf_cache/", step_name = None):
    # TODO add config as input, add batch size as config option
    setup_logger(logger, get_log_config())
    cache_name = f"{step_name}{str(year)}-{str(end_year)}"
    output_signature = (
        tf.TensorSpec(shape=(None,(n_back+1) * n_features), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),


    )

    if step_name is None:
        raise ValueError("Must give step name")
    if not list((Path(cache_dir).glob(f"{cache_name}*"))):
        logger.info(f"Cache not found for {step_name}, generating data")
        db_files = [str(x) for x in Path(db_dir).glob("*.parquet")]
        available_ids = duckdb.sql(f"SELECT id_OpenAlex FROM read_parquet({db_files})").df()
        id_index_map = make_id_index_map(db_dir, "id_OpenAlex")

        df = duckdb.sql(f"""SELECT id_OpenAlex, publication_date_int, referenced_works_OpenAlex 
        FROM read_parquet({db_files}) 
        WHERE publication_date_int >= {year} AND publication_date_int < {end_year}""").df()

        df = df.sort_values(sort_col, ascending=True)

        df["referenced_works_OpenAlex"] = df["referenced_works_OpenAlex"].apply(replace_ref_works)
        available_ids.columns = ["referenced_works_OpenAlex"] 
        df = df.explode("referenced_works_OpenAlex")
        df = df.merge(available_ids, on="referenced_works_OpenAlex", how="inner")
        
       # df = df.groupby("id_OpenAlex")
        embedding_cols = [f"embedding_{x}" for x in range(n_features)]
        y_df = duckdb.sql(f"""
        SELECT id_OpenAlex, higher_than_median_year 
        FROM read_parquet({db_files}) 
        WHERE publication_date_int >= {year} AND publication_date_int < {end_year}
        """).df()
        label_map = pd.Series(y_df.higher_than_median_year.values, index=y_df.id_OpenAlex).to_dict()
        del y_df

        dataset = tf.data.Dataset.from_generator(
        lambda : group_generator_pivot(Path(db_files[0]).parent, df, id_index_map, label_map, n_back, embedding_cols, batch_size=(1024*2)),
            output_signature=output_signature,
        )
        """
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
        """
        
        dataset = dataset.cache(os.path.join(cache_dir, cache_name))
        for _ in dataset:
            pass

        logger.info(f"Dataset {step_name} cached")

    else:
        dataset = tf.data.Dataset.from_generator(
        lambda : None,
            output_signature=output_signature
        )
        dataset = dataset.cache(os.path.join(cache_dir, cache_name))
        
        logger.info(f"Dataset {step_name} loaded")

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
   
    return dataset        

