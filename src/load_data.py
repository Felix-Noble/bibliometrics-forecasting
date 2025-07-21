#src/load_data.py
import dask.dataframe as dd
import pandas as pd

def get_sorted_id_ref_map(id_ref_list_df : dd.DataFrame, embedding_source_df):
    """
    Efficiently finds and aggregates referenced embeddings for each paper in a Dask DataFrame.

    Args:
        df (dd.DataFrame): DataFrame with columns 'id_OpenAlex' and 'referenced_works_OpenAlex' (list of IDs).
        embedding_source_df (dd.DataFrame): The main DataFrame containing all papers, embeddings,
                                             and a 'publication_date' column. 
                                             MUST be indexed by 'id_OpenAlex'.
    """
    # Ensure the source dataframe is indexed for a fast merge
    if not embedding_source_df.known_divisions:
        print("Setting index on embedding_source_df for efficient merging...")
        embedding_source_df = embedding_source_df.set_index("id_OpenAlex").persist()

    # 1. Explode the list of referenced works into a long-format DataFrame
    long_df = id_ref_list_df[['id_OpenAlex', 'referenced_works_OpenAlex']].explode('referenced_works_OpenAlex').persist()
   
    # 2. Perform a single, efficient merge to get all referenced data
    # We select only the columns we need from the source DataFrame
    source_cols = ['publication_date'] 
    
    merged_df = dd.merge(
        left=long_df,
        right=embedding_source_df[source_cols],
        left_on='id_OpenAlex',
        right_index=True,
        how='inner' # Use 'inner' to drop references that weren't found
    )
    del long_df
    # 3. Sort values before grouping to ensure embeddings are ordered by date
    merged_df["publication_date"] = dd.to_datetime(merged_df["publication_date"]).persist()

    merged_df = merged_df.sort_values(['publication_date','id_OpenAlex']).persist()
    merged_df = merged_df.set_index("publication_date").persist()
    # 4. Group by the original paper ID and aggregate embeddings into a list
    # Define a function to collect the embedding values into a list
    
    """embedding_cols = [col for col in sorted_df.columns if col.startswith('embedding_')]
    
    aggregated = sorted_df.groupby('id_OpenAlex').agg({
        col: lambda s: s.tolist() for col in embedding_cols
    })
    """

    return merged_df

# --- Example Usage ---
# result = get_all_referenced_embeddings(my_partitioned_df, whole_df)
# computed_results = result.compute()

"+ [col for col in embedding_source_df.columns if col.startswith('embedding_')]"