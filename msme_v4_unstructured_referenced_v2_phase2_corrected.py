import os
import base64
import time
from mistralai import Mistral
from dateutil import parser  # ensure parser is imported
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
    'PIB2040249.pdf': 'EMPLOYMENT OPPORTUNITIES IN MSMEs - PIB 01 AUG 2024',
    'PIB2040253.pdf': 'PUBLIC PROCUREMENT POLICY FOR MICRO AND SMALL ENTERPRISES PIB 01 AUG 2024',
    'PIB2040254.pdf': 'FINANCIAL ASSISTANCE FOR MSMEs - PIB 01 AUG 2024',
    'PIB2040257.pdf': 'PERFORMANCE OF KVIC - PIB 01 AUG 2024',
    'PIB2040258.pdf': 'MSMEs EXPORTERS - PIB 01 AUG 2024',
    'PIB2040261.pdf': 'PM VISHWAKARMA YOJANA - PIB 01 AUG 2024',
    'PIB2040262.pdf': 'FINANCIAL ASSISTANCE TO MSME TRADERS - PIB 01 AUG 2024',
    'PIB2040264.pdf': 'PROMOTION OF MSMEs TRADERS - PIB 01 AUG 2024',
    'PIB2040266.pdf': 'DEVELOPMENT OF MSME SECTOR - PIB 01 AUG 2024',
    'PIB2041681.pdf': 'STATUS ON THE IMPLEMENTATION OF AATMANIRBHAR BHARAT YOJANA - PIB 05 AUG 2024',
    'PIB2041682.pdf': 'Sub-schemes under the Central Sector Scheme(RAMP) - PIB 05 AUG 2024',
    'PIB2041684.pdf': 'CREDIT ACCESS AND FINANCE FOR MSMEs - PIB 05 AUG 2024',
    'PIB2041685.pdf': 'Initiatives to increase the participation of women in the MSMEs - PIB 05 AUG 2024',
    'PIB2041686.pdf': 'EXPORT OF MSMEs PRODUCTS - PIB 05 AUG 2024',
    'PIB2041687.pdf': 'AATMANIRBHAR BHARAT IN THE MSME SECTOR - PIB 05 AUG 2024',
    'PIB2041690.pdf': 'Measures to boost the MSME sector in the country - PIB 05 AUG 2024',
    'PIB2041699.pdf': '“Raising and Accelerating MSME Performance” (RAMP) Scheme - PIB 05 AUG 2024',
    'PIB2043186.pdf': 'PROCUREMENT OF GOODS AND SERVICES FROM MSMEs - PIB 08 AUG 2024',
    'PIB2043187.pdf': 'PM VISHWAKARMA SCHEME - PIB 08 AUG 2024',
    'PIB2043189.pdf': 'SUPPORT TO SMEs - PIB 08 AUG 2024',
    'PIB2043191.pdf': 'COIR PRODUCTION - PIB 08 AUG 2024',
    'PIB2043192.pdf': 'ENTREPRENEURSHIP AND SKILL DEVELOPMENT PROGRAMME - PIB 08 AUG 2024',
    'PIB2043895.pdf': 'Har Ghar Tiranga - PIB 09 AUG 2024',
    'PIB2044753.pdf': 'National Institute for MSME (NI-MSME) - PIB 13 AUG 2024',
    'PIB2045004.pdf': 'MoU signed between Ministry of MSME and Small Business Administration (SBA), Government of United States of America - PIB 13 AUG 2024',
    'PIB2045671.pdf': 'Tiranga Yatra - PIB 15 AUG 2024',
    'PIB2046003.pdf': 'E Tendering Program and Facilitation on Credit Support - PIB 16 AUG 2024',
    'PIB2047302.pdf': 'Memorandum of Understanding between KVIC and Department of Posts - PIB 21 AUG 2024',
    'PIB2048380.pdf': 'Har Ghar Tiranga Abhiyan - PIB 23 AUG 2024',
    'PIB2050360.pdf': 'Union Minister Shri Jitan Ram Manjhi reviews KVI sector performance and Khadi Mahotsav - PIB 31 AUG 2024',
    'PIB2079786.pdf': 'SCHEMES TO REVIEW MSME SECTOR FACING CRISIS - PIB 02 DEC 2024',
    'PIB2079789.pdf': 'EXPANSION OF MICRO, SMALL AND MEDIUM ENTERPRISES (MSMEs) - PIB 02 DEC 2024',
    'PIB2080700.pdf': 'NSIC - PIB 04 DEC 2024',
    'PIB2083806.pdf': 'Women Owned MSMEs - PIB 12 DEC 2024',
    'PIB2084812.pdf': 'MSE-SPICE Scheme - PIB 16 DEC 2024',
    'PIB2087361.pdf': 'The MSME Revolution Transforming India’s Economic Landscape - PIB 23 DEC 2024',
    'PIB2001816.pdf': 'Breaking records and reaching new heights - PIB 02 FEB 2024',
    'PIB2002569.pdf': 'Procurement of goods from SCs/STs and women entrepreneurs - PIB 05 FEB 2024',
    'PIB2002570.pdf': 'Development of MSME in the country - PIB 05 FEB 2024',
    'PIB2002571.pdf': 'Credit for Micro Small and Medium Enterprises - PIB 05 FEB 2024',
    'PIB2002572.pdf': 'Employment opportunities in MSMEs - PIB 05 FEB 2024',
    'PIB2002575.pdf': 'PM Vishwakarma Scheme - PIB 05 FEB 2024',
    'PIB2002577.pdf': 'Agro based industries in rural areas - PIB 05 FEB 2024',
    'PIB2003866.pdf': 'Indigenous Manufacturing of Medical Devices and Toys - PIB 08 FEB 2024',
    'PIB2003867.pdf': 'National SC/ST Hub Centres - PIB 08 FEB 2024',
    'PIB2003868.pdf': 'Assistance to Khadi Units - PIB 08 FEB 2024',
    'PIB2003869.pdf': 'MSMEs Run by Women - PIB 08 FEB 2024',
    'PIB2003870.pdf': 'Impact of Rising Imports on Domestic Industries - PIB 08 FEB 2024',
    'PIB2003871.pdf': 'Contribution of MSME Sector in GDP - PIB 08 FEB 2024',
    'PIB2005582.pdf': 'Shri Narayan Rane to inaugurate Technology Centres - PIB 13 FEB 2024',
    'PIB2006064.pdf': 'Shri Narayan Rane says MSMEs are the backbone of our economy and all possible efforts are being made to ensure that the MSMEs are duly supported - PIB 14 FEB 2024',
    'PIB2010341.pdf': 'Ministry of Micro, Small & Medium Enterprises organizes CPSE conclave - PIB 29 FEB 2024',
    'Women Entrepreneurs in MSMEs.pdf': 'Women Entrepreneurs in MSMEs - PIB 05 FEB 2024',
    'PIB1992725.pdf': 'SUCCESS STORY - PIB 03 JAN 2024',
    'PIB1992733.pdf': 'MoU signed between QCI and KVIC in Ahmedabad - PIB 03 JAN 2024',
    'PIB1992842.pdf': 'YEAR END REVIEW – 2023 - PIB 03 JAN 2024',
    'PIB1993606.pdf': 'Atmanirbhar Bharat Utsav 2024 - PIB 05 JAN 2024',
    'PIB1997098.pdf': 'Khadi Sanatan Vastra - PIB 17 JAN 2024',
    'PIB1998707.pdf': 'Consecration ceremony - PIB 23 JAN 2024',
    'PIB1999280.pdf': 'PM Vishwakarma beneficiaries invited as “special guests” to witness the Republic Day Parade - PIB 24 JAN 2024',
    'PIB1999495.pdf': 'Shri Narayan Rane interacts with PM Vishwakarma beneficiaries invited as “special guests” to witness the Republic Day Parade - PIB 25 JAN 2024',
    'Success Story- Enterprise ‘KARUANGAN’ Manufacturer and  Exporter – Dyeing and printing.pdf': 'Success Story KARUANGAN - PIB 15 JAN 2024',
    'Success Story-PMEGP loan helping in growth of Enterprise.pdf': 'Success Story-PMEGP loan helping in growth of Enterprise - PIB 03 JAN 2024',
    'PIB2031853.pdf': 'KVIC Sets New Record Under the Leadership of Prime Minister Shri Narendra Modi - PIB 09 JUL 2024',
    'PIB2034414.pdf': 'First Yashasvini Awareness Campaign organised in Jaipur - PIB 19 JUL 2024',
    'PIB2035068.pdf': 'VENDOR DEVELOPMENT PROGRAMMES FOR MSMEs - PIB 22 JUL 2024',
    'PIB2035073.pdf': 'CONTRIBUTION OF MSMEs TO THE GDP - PIB 22 JUL 2024',
    'PIB2035075.pdf': 'BLOCKCHAIN-POWERED SMART CONTRACTS FOR MSMEs - PIB 22 JUL 2024',
    'PIB2035077.pdf': 'TECHNOLOGICAL TRANSFORMATION OF MSMEs - PIB 22 JUL 2024',
    'PIB2035080.pdf': 'WOMEN-LED MSMEs - PIB 22 JUL 2024',
    'PIB2035082.pdf': 'OPEN NETWORK DIGITAL COMMERCE (ONDC) - PIB 22 JUL 2024',
    'PIB2035085.pdf': 'NEW SCHEME FOR MSMEs - PIB 22 JUL 2024',
    'PIB2035088.pdf': 'COLLABORATION BETWEEN INDUSTRIES AND ACADEMIA - PIB 22 JUL 2024',
    'PIB2035092.pdf': 'INTERNSHIPS FOR STUDENTS OF VARIOUS TECHNICAL DISCIPLINES WITHIN MSMEs - PIB 22 JUL 2024',
    'PIB2035095.pdf': 'DIGITAL TRANSFORMATION OF MSMEs - PIB 22 JUL 2024',
    'PIB2035097.pdf': 'PARTICIPATION OF MSMEs IN DIGITAL COMMERCE WITHOUT PAYING HIGH COMMISSION - PIB 22 JUL 2024',
    'PIB2035100.pdf': 'PRIME MINISTER’S EMPLOYMENT GENERATION PROGRAMME (PMEGP) - PIB 22 JUL 2024',
    'PIB2035103.pdf': 'CLIMATE CHANGE AWARENESS AND FINANCING FOR MSMEs - PIB 22 JUL 2024',
    'PIB2036982.pdf': 'VIKSIT BHARAT 2047 UNDER MSMEs - PIB 25 JUL 2024',
    'PIB2036985.pdf': 'EXPORT OF SMALL PRODUCTS - PIB 25 JUL 2024',
    'PIB2036989.pdf': 'IRON, STEEL FABRICATION AND CLOTH SECTOR - PIB 25 JUL 2024',
    'PIB2036997.pdf': 'KHADI AND VILLAGE INDUSTRIES COMMISSION - PIB 25 JUL 2024',
    'PIB2038536.pdf': 'EMPLOYMENT OPPORTUNITIES IN MSME - PIB 29 JUL 2024',
    'PIB2038537.pdf': 'INNOVATIONS IN MSMEs SECTOR - PIB 29 JUL 2024',
    'PIB2038539.pdf': 'MSME CLUSTER DEVELOPMENT PROGRAMME - PIB 29 JUL 2024',
    'PIB2038541.pdf': 'CREATION OF DIGITAL PUBLIC INFRASTRUCTURE - PIB 29 JUL 2024',
    'PIB2038542.pdf': 'STEPS TAKEN TO ENHANCE AND SIMPLIFY CREDIT FLOW TO MSMEs - PIB 29 JUL 2024',
    'PIB2038545.pdf': 'EMPLOYMENT GENERATION PROGRAMME - PIB 29 JUL 2024',
    'PIB2038546.pdf': 'AVAILABILITY OF RAW MATERIAL FOR MSME SECTOR - PIB 29 JUL 2024',
    'PIB2038549.pdf': 'FINANCIAL AND OTHER MAJOR ISSUES PLAGUING MSME SECTOR - PIB 29 JUL 2024',
    'Shri Jitan Ram Manjhi says MSME Sector is the backbone of the Indian economy and Empowering rural entrepreneurs would be the priority in order to ensure holistic growth of the MSME Sector.pdf': 'MSME Sector is the backbone of the Indian economy - PIB 18 JUL 2024',
    'JOBS IN MSME SECTOR.pdf': 'JOBS IN MSME SECTOR - PIB 29 JUL 2024',
    'EMPLOYMENT GENERATION IN MSMEs SECTOR.pdf': 'EMPLOYMENT GENERATION IN MSMEs SECTOR - PIB 25 JUL 2024',
    'PIB2024009.pdf': 'Union Minister Shri Jitan Ram Manjhi and Minister of State Sushri Shobha Karandlaje assume charge of the MSME - PIB 11 JUN 2024',
    'PIB2024047.pdf': 'Viksit Bharat Abhiyan - PIB 11 JUN 2024',
    'PIB2028582.pdf': 'Udyami Bharat - MSME Day event - PIB 25 JUN 2024',
    'PIB2028855.pdf': 'Bumper sale of Khadi yoga clothes and mats on International Yoga Day - PIB 26 JUN 2024',
    'PIB2029130.pdf': 'Shri Jitan Ram Manjhi says MSMEs will be a key force in the movement towards Atmanirbhar and Viksit Bharat - PIB 27 JUN 2024',
    'PIB2029464.pdf': 'Shri Jitan Ram Manjhi reviews performance of the schemes run by KVIC - PIB 29 JUN 2024',
    'PIB2010607.pdf': 'KVIC delivers training and distributes high-quality machinery and toolkits to rural artisans at Nashik - PIB 01 MAR 2024',
    'PIB2010753.pdf': 'Panjikaran se Pragati, WEP- Unnati-Udyamita se Pragati and Mentorship Platform - PIB 01 MAR 2024',
    'PIB2011262.pdf': 'Registrations of Informal Micro Enterprises on Udyam Assist Platform - PIB 04 MAR 2024',
    'PIB2012759.pdf': 'Shri Narayan Rane to lay the foundation stone of Multipurpose Exhibition-cum-Convention Centre at Saki Naka - PIB 08 MAR 2024',
    'PIB2013168.pdf': 'Shri Narayan Rane to lay the foundation stone of MSME-Technology Centre - PIB 10 MAR 2024',
    'PIB2013212.pdf': 'Collaboration on styling of DD News - PIB 10 MAR 2024',
    'PIB2013494.pdf': 'Shri Narayan Rane lays the foundation stone of MSME-Technology Centre - PIB 11 MAR 2024',
    'PIB2013568.pdf': 'KVIC Empowers Rural Artisans - PIB 11 MAR 2024',
    'PIB2014243.pdf': 'Shri Narayan Rane approves setting up of EAEC and COMET at National Institute for MSME - PIB 13 MAR 2024',
    'PIB2014815.pdf': 'Shri Bhanu Pratap Singh Verma lays the foundation stone for Coir Showroom at Konch - PIB 15 MAR 2024',
    'PIB2014988.pdf': 'Empowering Rural Artisans with Machinery and Toolkits - PIB 15 MAR 2024',
    'Shri Narayan Rane lays the foundation stone of Multipurpose Exhibition-cum-Convention Centre at Saki Naka, Mumbai.pdf': 'Shri Narayan Rane lays the foundation stone of Multipurpose Exhibition-cum-Convention Centre at Saki Naka - PIB 09 MAR 2024',
    'Total number of registered enterprises on Udyam and UAP crosses 4 crore, a major milestone for formalization initiative undertaken by the Ministry.pdf': 'Total number of registered enterprises on Udyam and UAP - PIB 15 MAR 2024',
    'PIB2070639.pdf': 'Honoring Artisans: PM Vishwakarma Yojana - PIB 04 NOV 2024',
    'PIB2073846.pdf': 'MSME PAVILION - PIB 16 NOV 2024',
    'PIB2073904.pdf': 'COIR Board PAVILION - PIB 16 NOV 2024',
    'PIB2073933.pdf': 'Khadi India Pavilion - PIB 16 NOV 2024',
    'PIB2075100.pdf': 'MSME PAVILION - PIB 20 NOV 2024',
    'PIB2075250.pdf': 'Khadi India and Coir Board Pavilions - PIB 20 NOV 2024',
    '11thmeeting.pdf': '11th Meeting of General Counsil of MGIRI - PIB 25 NOV 2024',
    'Ministry of Micro, Small & Medium Enterprises (MSME) successfully concludes Special Campaign 4.0 in a befitting manner.pdf': 'MSME successfully concludes Special Campaign 4.0 - PIB 06 NOV 2024',
    'PIB2061265.pdf': 'Wage Increase for Khadi Artisans and Special Discounts on Gandhi Jayanti - PIB 02 OCT 2024',
    'PIB2061457.pdf': 'Institutionalizing Swachhata, Minimizing Pendency - PIB 03 OCT 2024',
    'PIB2062088.pdf': 'Sales on Gandhi Jayanti - PIB 04 OCT 2024',
    'PIB2066166.pdf': 'National SC-ST hub - PIB 18 OCT 2024',
    'PIB2066264.pdf': 'Khadi Mahotsav - PIB 19 OCT 2024',
    'PIB2066306.pdf': 'The Fabric of Freedom, The Language of Fashion - PIB 19 OCT 2024',
    'Ministry of Micro, Small & Medium Enterprises (MSME) conducts ‘Special Campaign 4.0’ from 2nd to 31st October, 2024 with full vigour and enthusiasm.pdf': 'Special Campaign 4.0 - PIB 18 OCT 2024',
    'Shri Jitan Ram Manjhi, Union Minister for MSME led the cleanliness drive under campaign ‘Swachhata Hi Seva’.pdf': 'Swachhata Hi Seva - PIB 01 OCT 2024',
    'PIB2054694.pdf': 'Special Campaign 4.0 - PIB 13 SEP 2024',
    'PIB2054763.pdf': 'New Khadi of New India - PIB 13 SEP 2024',
    'PIB2055095.pdf': 'Inauguration of Centre for Rural Enterprise Acceleration through Technology at Leh - PIB 14 SEP 2024',
    'PIB2055841.pdf': 'SailaiSamridhiYojana - PIB 17 SEP 2024',
    'PIB2056827.pdf': 'Swachhata Hi Seva 2024 Campaign - PIB 19 SEP 2024',
    'Union Minister of MSME, Shri Jitan Ram Manjhi, briefed the media on 100 days’ achievements of the Government in MSME sector.pdf': 'Union Minister of MSME - PIB 17 SEP 2024',
    'PIB2089308.pdf': 'YEAR END REVIEW – 2024 MINISTRY OF MICRO, SMALL AND MEDIUM ENTERPRISES - PIB 01 JAN 2025',
    'PIB2091702.pdf': 'Sushri Shobha Karandlaje participated in the 18th Pravasi Bharatiya Diwas Convention at Janta Maidan - PIB 10 JAN 2025',
    'PIB2094126.pdf': 'Khadi and Village Industries Commission inaugurates National-level Khadi Exhibition at Mahakumbh 2025 - PIB 18 JAN 2025',
    'PIB2094315.pdf': 'ODOP Exhibition displayed in 6000 Square Meters at Mahakumbh 2025 - PIB 19 JAN 2025',
    'PIB2095434.pdf': 'PM Vishwakarma Scheme Beneficiaries Honored with Invitation to witness the Republic Day Parade 2025 at Kartavya Path - PIB 23 JAN 2025',
    'PIB2097198.pdf': 'Union Minister Shri Jitan Ram Manjhi administered a Pledge on TB Mukt Bharat - PIB 28 JAN 2025',
    'PIB2099687.pdf': 'Budget 2025-26: Fuelling MSME Expansion - PIB 04 FEB 2025',
    'PIB2106151.pdf': 'KVIC organizes state level PMEGP exhibition at PWD Ground - PIB 25 FEB 2025',
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
    raw_date = extract_date_from_reference(reference)
    date_str = normalize_to_month_year(raw_date)


    # Add the extracted date to the dates list
    dates.extend([date_str] * len(chunks))

    sources = [file_name] * len(chunks)
    categories = [category] * len(chunks)
    references = [reference] * len(chunks)

    # Create combined content string for better searchability
    max_length = 8192
    final_chunks = []
    final_page_numbers = []
    final_combined_contents = []
    final_embeddings = []
    final_sources = []
    final_categories = []
    final_references = []
    final_dates = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    for chunk, page in zip(chunks, page_numbers):
        combined = f'Content from {reference}. Page number: {page}. {chunk}'
        if len(combined) <= max_length:
            final_combined_contents.append(combined)
            final_chunks.append(chunk)
            final_page_numbers.append(page)
            final_embeddings.append(embedding_model.encode(combined))
            final_sources.append(file_name)
            final_categories.append(category)
            final_references.append(reference)
            final_dates.append(date_str)
        else:
            # Split chunk further
            sub_chunks = splitter.split_text(chunk)
            for sub_chunk in sub_chunks:
                combined_sub = f'Content from {reference}. Page number: {page}. {sub_chunk}'
                if len(combined_sub) <= max_length:
                    final_combined_contents.append(combined_sub)
                    final_chunks.append(sub_chunk)
                    final_page_numbers.append(page)
                    final_embeddings.append(embedding_model.encode(combined_sub))
                    final_sources.append(file_name)
                    final_categories.append(category)
                    final_references.append(reference)
                    final_dates.append(date_str)
                else:
                    print(f'Still too large even after splitting. Skipping problematic chunk from page {page}')

    if len(final_combined_contents) == len(final_page_numbers) == len(final_embeddings):
        try:
            collection.insert([
                final_sources,
                final_page_numbers,
                final_categories,
                final_embeddings,
                final_combined_contents,
                final_references,
                final_dates
            ])
            print(f'Loaded PDF: {file_name} | Reference: {reference} | Date: {date_str}')
            return True
        except ParamError as e:
            print(f'[SKIPPED] {file_name} | Reason: {e}')
            return False
    else:
        print(f'Error: Mismatch in data lengths after safe splitting for {file_name}')
        return False

# Helper to normalize dates
def normalize_to_month_year(date_str: str) -> str:
    # handle ranges (use ASCII hyphen)
    if '-' in date_str:
        parts = re.split(r'-', date_str)
        date_str = parts[-1].strip()
    try:
        dt = parser.parse(date_str, default=datetime(1900,1,1))
        return dt.strftime('%B %Y')
    except:
        return date_str

def extract_date_from_reference(reference):
    

    return "February 2025"


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
cpi_unstructured_folder = '/home/tata_user/Projects/VR/Milvus_search/Date'


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
    if process_pdf_mistral(pdf_file, 'MSME'):
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
