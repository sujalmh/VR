import os
import base64
import time
from mistralai import Mistral
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import connections, db, utility, FieldSchema, DataType, Collection, CollectionSchema
from dotenv import load_dotenv
import re
from datetime import datetime
from pymilvus.exceptions import ParamError
# Load environment variables
load_dotenv()

MISTRAL_OCR_KEY = os.getenv("MISTRAL_OCR_KEY")
client = Mistral(api_key=MISTRAL_OCR_KEY)

# Connect to Milvus
conn = connections.connect(host='localhost', port=19530)

# Ensure the database is created and selected
db.list_database()
if 'tata_db' not in db.list_database():
    db.create_database('tata_db')
db.using_database('tata_db')

# List collections to ensure they exist
utility.list_collections()

# Define schema for the Milvus collection
id_field = FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True)
source_field = FieldSchema(name='source', dtype=DataType.VARCHAR, max_length=255)
page_field = FieldSchema(name='page', dtype=DataType.INT64)
category_field = FieldSchema(name='category', dtype=DataType.VARCHAR, max_length=50)
embedding_field = FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, dim=768)
content_field = FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=8192)  # ← updated here
reference_field = FieldSchema(name='reference', dtype=DataType.VARCHAR, max_length=255)
date_field = FieldSchema(name='date', dtype=DataType.VARCHAR, max_length=50)  # Add the 'date' field

# Update schema to include the 'date' field
schema = CollectionSchema(fields=[
    id_field, source_field, page_field, category_field,
    embedding_field, content_field, reference_field, date_field  # Added date_field
])

# Create or load the new collection for cpi_unstructured
collection_name = 'cpi_v6'
if collection_name not in utility.list_collections():
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

# Define index parameters and create the index
index_params = {
    'metric_type': 'COSINE',
    'index_type': 'HNSW',
    'params': {
        'M': 16,
        'efConstruction': 200
    }
}
collection.create_index(field_name='embeddings', index_params=index_params)
collection.load()

# Load Sentence Transformer model
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', trust_remote_code=True)

# Reference mapping for filenames
reference_map = {
    'AccountsataGlance2019_2020.pdf': 'Accounts At A Glance 2019-20',
    'AccountsataGlance2020_2021.pdf': 'Accounts At A Glance 2020-21',
    'AccountsataGlance2021_2022.pdf': 'Accounts At A Glance 2021-22',
    'AccountsataGlance2022_2023.pdf': 'Accounts At A Glance 2022-23'
}




def process_pdf_mistral(pdf_file, category):
    page_markdown_list = extract_markdown_from_pdf(pdf_file)

    if not page_markdown_list:
        print(f'Error: Failed to load document {pdf_file}')
        return False

    chunks = []
    page_numbers = []
    dates = []  # List to store the extracted dates

    for page_number, markdown in page_markdown_list:
        page_chunks = content_aware_chunk(markdown, chunk_size=2500, chunk_overlap=200)
        chunks.extend(page_chunks)
        page_numbers.extend([page_number] * len(page_chunks))  # Track which chunk came from which page

    if not chunks:
        print(f'Error: No content extracted from {pdf_file}')
        return False

    file_name = os.path.basename(pdf_file)
    reference = reference_map.get(file_name, 'Unknown Reference')

    # Extract month and year from the reference (e.g., "Minutes of the Monetary Policy Committee Meeting December 2023")
    date_str = extract_date_from_reference(reference)

    # Add the extracted date to the dates list
    dates.extend([date_str] * len(chunks))

    sources = [file_name] * len(chunks)
    categories = [category] * len(chunks)
    references = [reference] * len(chunks)

    # Create combined content string for better searchability
    combined_contents = [
        f'Content from {reference}. Page number: {page}. {chunk}'
        for chunk, page in zip(chunks, page_numbers)
    ]

    content_embeddings = [embedding_model.encode(content) for content in combined_contents]

    if len(sources) == len(page_numbers) == len(content_embeddings) == len(combined_contents) == len(categories) == len(references) == len(dates):
        try:
            collection.insert([
                sources, page_numbers, categories, content_embeddings, combined_contents, references, dates
            ])
            print(f'Loaded PDF: {file_name} | Reference: {reference} | Date: {date_str}')
            return True
        except ParamError as e:
            print(f'[SKIPPED] {file_name} | Reason: {e}')
            return False
    else:
        print(f'Error: Mismatch in data lengths for {file_name}')
        return False

def extract_date_from_reference(reference):
    # You can use regular expressions to extract the month and year from the reference string
    # For example, assuming the format is something like "December 2023"
    # Check for a time period like 2019-20 or 2021–2022
    reference_lower = reference.lower()
    match = re.search(r'(\d{4})[\u2013\-](\d{2}|\d{4})', reference)
    if match:
        start_year = int(match.group(1))
        end_year_raw = match.group(2)
        end_year = (
            int(end_year_raw) if len(end_year_raw) == 4
            else start_year // 100 * 100 + int(end_year_raw)
        )
        return f'January {start_year} - January {end_year}'

    # Match month and year like "February 2020"
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    for month in months:
        pattern = rf'\b{month.lower()}\s+(\d{{4}})\b'
        match = re.search(pattern, reference_lower)
        if match:
            return f'{month} {match.group(1)}'

    return "Unknown Date"

# === Extract markdown from a single PDF ===
def extract_markdown_from_pdf(file_path):

    if not file_path:
        print(f'Error: Failed to load document {pdf_file}')
        return False

    with open(file_path, "rb") as f:
        encoded_pdf = base64.b64encode(f.read()).decode("utf-8")
    document = {
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{encoded_pdf}"
    }
    try:
        ocr_response = client.ocr.process(model="mistral-ocr-latest", document=document, include_image_base64=False)
        time.sleep(1)  # prevent rate limiting
        pages = ocr_response.pages if hasattr(ocr_response, "pages") else ocr_response
        return [(i + 1, page.markdown) for i, page in enumerate(pages)]

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

# === Content-aware chunking ===
def content_aware_chunk(text, chunk_size=2500, chunk_overlap=200):
    # Step 1: Tag section headers
    #text = re.sub(r'(?m)^(\d+)\.\s+', r'[NUMBERED] \1. ', text)  # Tag numbered sections
    text = re.sub(r'## (.*?)\n', r'[SECTION] \1\n', text)        # Markdown subheadings
    text = re.sub(r'# (.*?)\n', r'[SECTION] \1\n', text)         # Markdown headings
    #text = re.sub(r'\n\[SECTION\]', r'[SECTION]', text)          # Clean stray newlines

    # Step 2: Split text at each marker
    chunks = re.split(r'(?=\[SECTION\])', text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Step 3: Apply RecursiveCharacterTextSplitter on each chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    final_chunks =  chunks
    #for chunk in chunks:
        #sub_chunks = splitter.split_text(chunk)
        #final_chunks.extend(sub_chunks)

    return final_chunks

# Folder path for PDFs
cpi_unstructured_folder = '/home/tata_user/Projects/VR/Milvus_search/Accounts_At_A_Glance'

pdf_files = [os.path.join(cpi_unstructured_folder, file)
             for file in os.listdir(cpi_unstructured_folder)
             if file.endswith('.pdf')]

failed_pdfs = []
total_files = len(pdf_files)
print(f'Total files to process: {total_files}')




start_time = time.time()
for idx, pdf_file in enumerate(pdf_files):
    print(f'Loading PDF: {os.path.basename(pdf_file)}')
    file_start_time = time.time()
    if process_pdf_mistral(pdf_file, 'financial_management_reports'):
        elapsed_time = time.time() - file_start_time
        print(f'Loaded PDF: {os.path.basename(pdf_file)} | Progress: {idx + 1}/{total_files} | Time elapsed: {elapsed_time:.2f} seconds')
    else:
        failed_pdfs.append(pdf_file)
    print('------------')

total_time = time.time() - start_time
print(f'All PDFs in the folder "{cpi_unstructured_folder}" have been processed and loaded into Milvus.')
print(f'Total time elapsed: {total_time:.2f} seconds.')

if failed_pdfs:
    print(f'The following PDFs failed to load into the Milvus collection: {failed_pdfs}')
else:
    print('All PDFs were successfully loaded into the Milvus collection.')
