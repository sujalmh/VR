import streamlit as st
from pymilvus import MilvusClient


@st.cache_resource
def get_milvus_client(uri: str, token: str = None) -> MilvusClient:
    client = MilvusClient(uri=uri, token=token)
    client.using_database("tata_db")  # Switch to tata_db
    return client


def create_collection(
    milvus_client: MilvusClient, collection_name: str, dim: int, drop_old: bool = True
):
    if milvus_client.has_collection(collection_name) and drop_old:
        milvus_client.drop_collection(collection_name)
    if milvus_client.has_collection(collection_name):
        raise RuntimeError(
            f"Collection {collection_name} already exists. Set drop_old=True to create a new one instead."
        )
    return milvus_client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="COSINE",
        consistency_level="Strong",
        auto_id=True,
    )


def get_search_results(milvus_client, collection_name, query_vector, output_fields=["id", "source", "page", "content", "reference", "date"],
                       date_filter = None):     # e.g., "2024-12-31"):
    # Build filter expression
    #start_date = "December 2023"
    #end_date = "February 2024"
    #filters = []
    #if start_date:
    #    filters.append(f'date >= "{start_date}"')
    #if end_date:
    #    filters.append(f'date <= "{end_date}"')
    #filter_expr = " and ".join(filters) if filters else None
    #filter_expr = '''date == "December 2023" or date == "January 2024" or date == "February 2024"'''

    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=50,
        search_params={"metric_type": "COSINE", "params": {'M': 16,
        'efConstruction': 200}},  # Using COSINE metric for embeddings
        output_fields=output_fields, # Use valid field names here
        group_by_field='reference',
        group_size=4,
        strict_group_size=False,
        filter=date_filter
    )
    return search_res