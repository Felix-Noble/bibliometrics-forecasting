from src.stream_parquet import DB_Pyarrow
from src.utils.setup_logging import setup_logger
from src.utils.load_config import get_data_config, get_log_config
import tensorflow as tf
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pandas as pd
import pyarrow as pa
import logging
import math
import os
import shutil
import json
logger = logging.getLogger(__name__)
setup_logger(logger, get_log_config())

def generate_parquet_timeseries(
    val_start,
    val_end,
    data_config: dict = get_data_config(),
    feature_dtype:type = pa.float32,
                              ):
    sort_col = data_config["sort_col"]
    id_col = data_config["id_col"]
    n_features = data_config["n_features"] 
    y_cols = data_config["y_cols"]
    feature_cols = [data_config["feature_col_name"] + str(x) for x in range(n_features)] 
    max_timepoints = data_config["max_timepoints"]
    pad_value = data_config["pad_value"]

    full_cols = pd.MultiIndex.from_product([feature_cols, range(max_timepoints)], names=[None, "ref_rank"])

    DB = DB_Pyarrow(dir = data_config["database_loc"],
                    id_col = data_config["id_col"],
                    RAM_MAX = data_config["RAM_MAX"],
                    )
    DB.batch_size = 1e6
    all_ids = DB.get_cols_all([id_col])
    all_ids_array = all_ids.to_table()[id_col]

    example_nbytes = (max_timepoints+1) * n_features * (feature_dtype().bit_width // 8) * 32
    referenced_works = DB.get_cols_range(cols = [id_col, "referenced_works_OpenAlex"],
                                          pyrange_min = val_start,
                                          pyrange_max = val_end,
                                         range_col = "publication_date_int",
                                        )
    referenced_works_table = referenced_works.to_table()
    source_ids_in_range = referenced_works_table[id_col].to_pandas()

    ref_works_lists = referenced_works_table["referenced_works_OpenAlex"]
    # Explode ref works lists
    exploded = pc.list_flatten(ref_works_lists)

    # Filter Ref Works on availalbe ids
    exploded_mask = pc.is_in(exploded, all_ids_array)
    exploded = pc.filter(exploded, exploded_mask)

    # Get exploded indices
    source_paper_ids = pc.list_parent_indices(ref_works_lists)
    source_paper_ids = pc.filter(source_paper_ids, exploded_mask)

    sort_col_table = DB.DB.scanner(
        filter = (ds.field(id_col).isin(exploded.to_pylist())),
        columns = [id_col, sort_col],
    ).to_table() 
    sort_col_table = DB.get_cols_all([id_col, sort_col]).to_table()
    ref_works_sort_col_i = pc.index_in(exploded, sort_col_table[id_col])

    exploded_table = pa.table({
        id_col : pc.take(referenced_works_table[id_col], source_paper_ids),
        "referenced_works_OpenAlex": exploded,
        "referenced_works_" + sort_col: pc.take(sort_col_table[sort_col], ref_works_sort_col_i)
    })
    
    batch_size = DB.get_batch_size(example_nbytes, verbose = 0)
    n_batches = math.ceil(source_ids_in_range.shape[0] / batch_size)
    for batch_i in range(n_batches):
        DB.get_mem_use(verbose = 0)
        batch_ids_table = source_ids_in_range.iloc[batch_i * batch_size : (batch_i+1) * batch_size]
        batch_ids_table = pa.Table.from_pandas(batch_ids_table.to_frame())

        exploded_batch_mask = pc.is_in(exploded_table[id_col], batch_ids_table[id_col])
        exploded_batch = pc.filter(exploded_table, exploded_batch_mask)

        # Get Features
        features_table = DB.get_cols_isin(cols = [id_col] + feature_cols,
                                    isin_array = exploded_batch["referenced_works_OpenAlex"],
                                    isin_col = id_col,
                                    ).to_table()
                                
        # Load into memory
        exploded_df = exploded_batch.to_pandas()
        features_df = features_table.to_pandas()

        # Merge features to exploded_df
        examples_df = exploded_df.merge(features_df, 
                                        right_on = id_col,
                                        left_on = "referenced_works_OpenAlex",
                                        how = "inner",
                                        )
        del exploded_df, features_df

        # Sort examples
        examples_df = examples_df.sort_values("referenced_works_" + sort_col, 
                                              ascending = True)
        # Get Ref Ranks
        examples_df["ref_rank"] = examples_df.groupby(id_col).cumcount()
        
        # Drop unused datapoints
        examples_df = examples_df[examples_df["ref_rank"] < max_timepoints]

        # Concatenate timeseries examples
        examples_df = examples_df.pivot_table(index = id_col,
                                              columns = "ref_rank",
                                              values = feature_cols,
                                 )
        if examples_df.shape[1] != max_timepoints * n_features:
            examples_df = examples_df.reindex(columns=full_cols)
        # Reorder
        examples_df.columns = [f'{col[0]}_{col[1]}' for col in examples_df.columns]
        examples_df = examples_df[[f'embedding_{x}_{y}' for y in range(max_timepoints-1, -1, -1) for x in range(n_features)]]

        # Get source_paper features
        source_features = DB.get_cols_isin(cols = [id_col, "publication_date_int"] + feature_cols + y_cols,
                                         isin_array = batch_ids_table[id_col],
                                         isin_col = id_col,
                                         ).to_table().to_pandas()
        if source_features.index.name == id_col:
            source_features = source_features.reset_index()
        if examples_df.index.name == id_col:
            examples_df = examples_df.reset_index()

        examples_df = examples_df.merge(source_features,
                                        right_on = id_col,
                                        left_on = id_col,
                                        how = "outer",
                                        )
        # Pad 
        examples_df = examples_df.fillna(pad_value)
        yield examples_df[[x for x in examples_df.columns if "embedding" in x]].values, examples_df[y_cols].values.reshape(-1, len(y_cols))

def create_tfrecord_example(features: list, target: list):
    """Creates tf_record example with timestamp (int) serialised_features (float > bytearray) and targets (int)"""
    
    feature = {
        #'timestamp': tf.train.Feature(int64_list = tf.train.Int64List(value = [timestamp])),
        'features': tf.train.Feature(float_list = tf.train.FloatList(value = features)),
        'target': tf.train.Feature(float_list = tf.train.FloatList(value = target)),
    }
    return tf.train.Example(features = tf.train.Features(feature = feature))

def process_tf_dataset(
    data_generator,
    t_min: int,
    t_max: int,
    data_config: dict = get_data_config(),
                    ):
    max_timepoints = data_config["max_timepoints"]
    n_features = data_config["n_features"]

    feature_cols = [f"embedding_{x}_{y}" for y in range(max_timepoints-1, -1, -1) for x in range(n_features)] + [f"embedding_{x}" for x in range(n_features)]
    target_cols = data_config["y_cols"]
    output_dir = data_config["tf_dataset_loc"]
    os.makedirs(output_dir, exist_ok = True)
    metadata = {"n_examples":{}}
    for shard_num, df_batch in enumerate(data_generator):
        n_examples = 0
        filename = os.path.join(output_dir, f"{t_min}_{t_max}_shard_{shard_num}.tfrecord")
        with tf.io.TFRecordWriter(filename) as writer:
            for _, row in df_batch.iterrows():
                features_tensor = row[feature_cols].to_list()
                #features_tensor = tf.reshape(features_tensor, (max_timepoints + 1, n_features)) 
                target = row[target_cols].to_list()
                #timestamp = tf.constant(row["publication_date_int"], dtype=tf.int64)
                
                example = create_tfrecord_example(features_tensor, target)
                writer.write(example.SerializeToString())
                n_examples += 1

        metadata_file = os.path.join(output_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

        metadata["n_examples"][filename] = n_examples
        logger.info(f"Saved Shard {shard_num} with {n_examples} examples to {filename}")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

def save_tf_dataset(data_config: dict = get_data_config()):
    if data_config["overwrite_tf_dataset"]:
        shutil.rmtree(data_config["tf_dataset_loc"])

    val_start = data_config["range_vals"]["publication_date_int_min"]
    val_end = data_config["range_vals"]["publication_date_int_max"]
    delta = data_config["range_vals"]["publication_date_int_delta"]

    for t_min in range(val_start, val_end, delta):
        val_start_int = pd.to_datetime(t_min, format="%Y").value
        val_end_int = pd.to_datetime(t_min + delta, format="%Y").value
        
        data_generator = generate_parquet_timeseries(val_start = val_start_int,
                                                     val_end = val_end_int)

        process_tf_dataset(data_generator = data_generator,
                        t_min = val_start_int,
                        t_max = val_end_int,
                        )

def tf_dataset_from_generator(
        val_start:int,
        val_end: int,
        data_config: dict = get_data_config()):
    max_timepoints = data_config["max_timepoints"]
    n_features = data_config["n_features"]
    y_cols = data_config["y_cols"]

    output_signature = (
        tf.TensorSpec(shape = (None, (max_timepoints+1)*n_features), dtype = tf.float32),
        tf.TensorSpec(shape = (None, len(y_cols)), dtype = tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: generate_parquet_timeseries(val_start, val_end),
        output_signature = output_signature,
    )
    return dataset

if __name__ == "__main__":
    main()



        

