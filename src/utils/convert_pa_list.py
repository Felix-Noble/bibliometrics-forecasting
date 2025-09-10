import pyarrow as pa
import pyarrow.parquet as pq
import ast
import pandas as pd
import dask.dataframe as dd
from dask.distributed import progress, Client

def convert_string_to_list_parquet(input_path: str, output_path: str, n_features = 384):
    """Convert existing string-based parquet to list-based"""
    
    # Read existing data
    table = pq.read_table(input_path)
    df = table.to_pandas()
    print("table read") 
    # Convert string column to actual lists
    df['referenced_works_OpenAlex'] = df['referenced_works_OpenAlex'].apply(
        lambda x: [int(id.replace("https://openalex.org/W", "")) 
                   for id in ast.literal_eval(x)] if pd.notna(x) else []
    )
    
    # Define explicit schema with list type
    schema = pa.schema([
        pa.field('id_OpenAlex', pa.int64()),
        pa.field('referenced_works_OpenAlex', pa.list_(pa.int64())),
        pa.field('publication_date_int', pa.int64()),
        # ... other fields
    ] + [pa.field(f'embedding_{i}', pa.float32()) for i in range(n_features)])
    
    # Convert to Arrow table with proper schema
    arrow_table = pa.Table.from_pandas(df, schema=schema)
    print("writing table")
    # Write with list compression
    pq.write_table(
        arrow_table, 
        output_path,
        compression='snappy',  # Good for list data
        use_dictionary=['referenced_works_OpenAlex']  # Can help with compression
    )

if __name__ == "__main__":

    n_features = 384
    db_loc = '~/AbstractTransformer/data/ACS'
    output = '~/AbstractTransformer/data/ACS_convert'
    client = Client(n_workers = 1, threads_per_worker = 1) 
    ddf = dd.read_parquet(db_loc)
    # Define schema upfront
    schema = pa.schema([
        pa.field('id_OpenAlex', pa.int64()),
        pa.field('referenced_works_OpenAlex', pa.list_(pa.int64())),
        pa.field('publication_date_int', pa.int64()),
        pa.field('higher_than_median_year', pa.int32()),
    ] + [pa.field(f'embedding_{i}', pa.float32()) for i in range(n_features)])
   
    # Convert string to list in each partition
    def convert_partition(partition):
        partition['referenced_works_OpenAlex'] = partition['referenced_works_OpenAlex'].apply(
            lambda x: [int(id.replace("https://openalex.org/W", "")) 
                       for id in ast.literal_eval(x)] if pd.notna(x) else []
        )
        return partition
    
    # Apply conversion
    df_converted = ddf.map_partitions(convert_partition, meta=ddf._meta)
    
    # Save with explicit schema
    write_task = df_converted.to_parquet(
        output,
        engine='pyarrow',
        schema=schema,
        compression='snappy',
        compute = False
    )
    print(df_converted.columns)
    progress(client.compute(write_task))
                                   
