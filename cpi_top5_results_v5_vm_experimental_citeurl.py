import logging
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from encoder import emb_text, model
from milvus_utils_crossencoder_v5 import get_milvus_client, get_search_results
import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from dateutil.relativedelta import relativedelta
from textwrap import dedent
from google import genai
from google.genai import types
from   google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import time
from time import strftime, gmtime
import re
from typing import List, Dict
# Load environment variables
load_dotenv()

# API Key and Security
API_KEY = os.getenv("ACQ_API_KEY")
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# FastAPI instance
app = FastAPI(title="Version 5 Server for Top 5 Search Results with URL and Summary")

# Milvus Configuration
CPI_V5_COLLECTION_NAME = os.getenv("CPI_V5_COLLECTION_NAME")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
TOP_N_RESULTS = 5  # Configurable number of search results

print(f'MILVUS_ENDPOINT = {MILVUS_ENDPOINT}')
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# Milvus client
milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)

# Logging setup
logging.basicConfig(
    filename="cpi_top5_results_output_v5_vm_experimental_citeurl.log",  # Log file name
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# API Key verification dependency
async def verify_api_key(api_key: str = Depends(api_key_header)):
    logging.info(f"Received API Key: {api_key[:4]}****")
    if api_key != os.getenv("ACQ_API_KEY"):
        logging.warning(f"Unauthorized API access attempt with key: {api_key[:4]}****")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Input model
class Question(BaseModel):
    question: str

def clarify_query(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""You are tasked with rephrasing the given query to make it easier for a RAG agent to pull the right data.
                Rules to produce rephrased query:
                    1. Do not edit the query as far as possible, only augment with a date range if none is present.
                        a. Remember that the current date is {curdate}.
                        b. If no date range is available from the context or from web search, use three months before {curdate} to {curdate}.
                    2. If the query contains acronyms, include full form in parentheses.
                        Example: Query -> What is RBI's thought process in May 2025?
                                Rephrased query -> What is RBI (Reserve Bank of India) thought process in May 2025?
                    3. Attach date range if no date range is available.
                        a. Include minimum and maximum date as far as possible (e.g. Feb 2025 to April 2025)
                        b. Always use month names and full years (e.g. April 2025, January 2022)
                    4. Do not attach extraneous information apart from this, or include your own thinking traces. Keep it as close to the original query as possible.
                    5. You MUST restrict your output to at most 25 words.

            The original query is given below.
            """),
            tools=[google_search_tool],
            temperature=0.0,
            ),
        contents=query
    )

    return response.text

"""
def clarify_query(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(fYou are tasked with rephrasing the given query to make it easier for a vector agent to pull the right data. ONLY IF NECESSARY, generate a rephrased natural language version of the query keeping in mind the following steps:
            1. Analyze the provided query for key entities. For example, if the query is about inflation in vegetables which is a type of food, then include the phrase "inflation in vegetables which is a type of food".
            2. Extract key phrases from the query and rewrite them with emphasis. For example, if the query asks about "state-level growth", rewrite it to "STATE LEVEL GROWTH (GDP)".
            3. Analyze the query for the existence of dates or time-related phrases.
                a. For example, if the query asks about GDP growth since December 2024, rephrase it as "GDP growth from [December 2024] to [{curdate}]".
                b. For example, if the query asks about IIP in the last six months, compute the date six months BEFORE [{curdate}] and add this information to the query.
                c. If the query asks about the last quarter, compute the last full quarter BEFORE [{curdate}] and add this information to the query.
                d. If the query asks about vague timelines such as "long term" without specifying dates, use the date five years ago from [{curdate}].
                e. IMPORTANT: If no date exists in the query use a range from 6 months before [{curdate}] to [{curdate}].
                f. If the query contains an event without a date (for example, "COVID" or "51st meeting" or "the last world cup"), then use the google_search_tool to attach a date to the event.
                VERY IMPORTANT: DO NOT use the web search to add any extraneous information to the query, apart from the extracted date.
                g. If the date is after [{curdate}], use [{curdate}] instead.
            4. If there are any numeric counters such as 1st, 2nd, 3rd, rewrite it in words. For example, 51st should be rewritten as "fifty first".
            5. Always remember that all queries are related to India. If the word "India" is not mentioned in the query, include this in the rephrased query.
            6. Do not include your reasoning trace in the rephrased query. Your output should contain only the information that is required from the Vector database.
            7. You MUST restrict your output to at most 25 words.

            tools=[google_search_tool],
            temperature=0.0,
            ),
        contents=query
    )

    return response.text
"""

def fetch_date(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""You are a date extractor. Your job is to extract a clear time reference point in time from user queries, if one exists.
            Examples:
            - "CPI report for December 2024" → "December 2024"
            - "What was the inflation rate in June 2023?" → "June 2023"
            - "Give me the latest IIP data" → "today"
            - "Tell me the GDP growth over the last five years" → "today"
            - "What happened in Q3 2022?" → "December 2022"
            - "What was the inflation rate H1 FY25?" → "September 2024"
            Return only the extracted date phrase (like "December 2024" or "today"). If no specific date is found, assume "today".
            IMPORTANT: If the extracted date contains multiple dates, output ONLY the LATEST date mentioned in the query.
            Make sure the output format of the date is "%B %Y"
            IMPORTANT: Do not output a date beyond {curdate}
            """),
            temperature=0.0,
        ),
        contents=query
    )

    query_date = response.text.strip()
    return query_date

def fetch_min_date(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""You are a date extractor. Your job is to extract a clear time reference point in time from user queries, if one exists.
            Examples:
            - "CPI report for December 2024" → "December 2024"
            - "What was the inflation rate in June 2023?" → "June 2023"
            - "Give me the latest IIP data" → "today"
            - "Tell me the GDP growth over the last five years" → "today"
            - "What happened in Q3 2022?" → "December 2022"
            - "What was the inflation rate H1 FY25?" → "September 2024"
            Return only the extracted date phrase (like "December 2024" or "today"). If no specific date is found, assume "today".
            IMPORTANT: If the extracted date contains multiple dates, output ONLY the EARLIEST date mentioned in the query.
            Make sure the output format of the date is "%B %Y"
            IMPORTANT: Do not output a date beyond {curdate}
            """),
            temperature=0.0,
        ),
        contents=query
    )

    query_date = response.text.strip()
    return query_date


def months_since(date_str, query_date='today'):
    #Date from the query if not use latest
    try:
        date_obj = datetime.strptime(date_str, '%B %Y')
        if query_date == 'today':
            today = datetime.today()
        else:
            try:
                today = datetime.strptime(query_date, '%B %Y')
            except:
                today = datetime.today()  # fallback

        diff = relativedelta(today, date_obj)
        logging.info("Pulled date: " + query_date + ", Doc date: " + date_str + ", Delta: " + str(diff.years * 12 + diff.months))
        return diff.years * 12 + diff.months

    except Exception as e:
        print(f'Error parsing date: {e}')
        return 999

def build_range_around_date(center_date_str, months_before, months_after, field_name="date"):
    """
    Given a center date string like 'March 2024' and two integers (months_before, months_after),
    builds a Milvus filter expression for that range.

    Assumes Milvus stores dates in the format 'Month YYYY' (e.g., 'March 2024').
    """
    if center_date_str == 'today':
        center_date_str = datetime.today().strftime("%B %Y")
    try:
        center_date = datetime.strptime(center_date_str, "%B %Y")
    except:
        center_date_str = datetime.today().strftime("%B %Y")
        center_date = datetime.strptime(center_date_str, "%B %Y")

    # Calculate start and end of the range
    start_date = center_date - relativedelta(months=months_before)
    end_date = center_date + relativedelta(months=months_after)

    # Generate the list of months
    current = start_date
    months = []
    while current <= end_date:
        months.append(current.strftime("%B %Y"))
        current += relativedelta(months=1)

    # Build the filter expression
    filters = [f'{field_name} == "{m}"' for m in months]
    filter_expr = " or ".join(filters)

    return {"filter": filter_expr}

def generalize_query(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent("""Consider the query given in the content. Your task is to generalize the query to a small extent. You can do this by:
            1. Removing mentions of specific states of India and replacing it by just "India". For example, "Tamil Nadu" can be replaced by "India".
            2. Removing specific day-month references and retaining only the years. For example, "23 May 2023" can be replaced by just "2023".
            3. Removing names of specific commodities and replacing them by broader classes. For example, "moong dal" can be replaced by "pulses" or "food".
            4. Removing generic phrases and focusing only on the data. For example, "Effect of COVID on steel production" can be replaced by "steel production statistics". Similarly, phrases such as "commentary", "effect", "policy" can be removed from the rephrased query.
            INSTRUCTIONS: Return the rephrased query as a single sentence. Do not hallucinate non-existing information. Stick to a maximum length of 12 words.
            """),
            temperature=0.0,
            ),
        contents=query
    )
    logging.info("Rephrased query: " + response.text)
    return response.text

def suggest_answer(query, excerpts):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""Consider the following query: {query}. You are given the following content that contains a potential answer for this query. Write a short paragraph or set of bullet points that summarize the answer to the query.

            Here is some brief context about aspects of the Indian economy:
                1. Agriculture: Contributes about 15-20% to GDP, agriculture employs nearly half of the workforce. Monsoon patterns significantly impact agricultural output and rural demand.
                2. Industry: Manufacturing and construction are crucial for GDP growth and employment. Government initiatives like "Make in India" aim to boost manufacturing.
                3. Services: The services sector, including IT, finance, and telecommunications, contributes over 50% to GDP. IT services, in particular, are a major export and growth driver.
                4. Government Policies: Fiscal policies, such as taxation and public spending, influence economic growth. Monetary policies by the Reserve Bank of India (RBI) manage inflation and interest rates.
                5. Inflation: Influenced by food prices, fuel costs, and global commodity prices. The RBI uses repo rates and cash reserve ratios to control inflation.
                6. Foreign Direct Investment (FDI): FDI inflows boost infrastructure, technology, and job creation. Government policies aim to attract FDI in sectors like manufacturing and services.
                7. Global Economic Conditions: Exports, remittances, and foreign investments are affected by global demand and economic stability.
                8. Demographics: A young and growing workforce can drive economic growth, but requires adequate education and employment opportunities.
                9. Infrastructure: Investments in transportation, energy, and digital infrastructure enhance productivity and economic growth.
                10. Technological Advancements: Innovation and digitalization improve efficiency and competitiveness across sectors.

            IMPORTANT: Include ONLY the information that answers the query. If there is insufficient information to answer the specific query, say exactly the following in your response: "<insufficient-data>". Do not include any other text when there is insufficient information to answer the query.
            **Formatting instructions**
            - Make your answer about 300 to 350 words, IF sufficient data is present.
            - Use each of the excerpts to compose your answer, so long as they are relevant to the query.
            - Do not mention any data that is not available or not present. Stick to a summary of what data is available.

            IMPORTANT: In your summary, focus on specific data values, prices, and percentages if they relate to the original query. Avoid returning purely qualitative observations.
            IMPORTANT: If you are returning a full answer, do not include "<insufficient-data>" in the response.
            IMPORTANT: DO NOT HALLUCINATE ANY INFORMATION THAT IS NOT PRESENT IN THE ATTACHED CONTENTS.
            """),
            temperature=0.0,
            ),
        contents=excerpts
    )
    return response.text

def get_reference_url(ref: str) -> str:
    if re.match(r"Inflation Expectations Survey of Households \w+ \d{4}", ref):
        return "https://website.rbi.org.in/web/rbi/statistics/survey?category=24927098&categoryName=Inflation%20Expectations%20Survey%20of%20House-holds%20-%20Bi-monthly"
    elif re.match(r"Monetary Policy Report \w+ \d{4}", ref):
        return "https://website.rbi.org.in/web/rbi/publications/articles?category=24927873"
    elif re.match(r"Minutes of the Monetary Policy Committee Meeting \w+ \d{4}", ref):
        return "https://website.rbi.org.in/web/rbi/press-releases?q=%22Minutes+of+the+Monetary+Policy+Committee+Meeting%22"
    elif re.match(r"CPI Press Release \w+ \d{4}", ref):
        return "https://www.mospi.gov.in/archive/press-release?field_press_release_category_tid=120"
    elif re.match(r"Economic Survey \d{4} ?- ?\d{4}", ref):
        return "https://www.indiabudget.gov.in/economicsurvey/allpes.php"
    elif re.match(r"IIP Press Release \w+ \d{4}", ref):
        return "https://www.mospi.gov.in/archive/press-release?field_press_release_category_tid=121"
    elif re.match(r"Monthly Economic Report \w+ \d{4}", ref):
         return "https://dea.gov.in/monthly-economic-report-table"
    elif re.match(r"RBI Bulletin \w+ \d{4}", ref):
         return "https://rbi.org.in/Scripts/BS_ViewBulletin.aspx"
    elif re.match(r"RBI State Finances \w+ \d{4}", ref):
         return "https://rbi.org.in/Scripts/AnnualPublications.aspx?head=State%20Finances%20:%20A%20Study%20of%20Budgets"
    elif re.match(r"RBI Handbook of Statistics On Indian States \w+ \d{4}", ref):
         return "https://rbi.org.in/Scripts/AnnualPublications.aspx?head=Handbook+of+Statistics+on+Indian+States"
    elif re.match(r"RBI Publications - Annual \d{4}", ref):
         return "https://rbi.org.in/Scripts/Publications.aspx?publication=Annual"
    elif re.match(r"RBI Publications - Half Yearly \w+ \d{4}", ref):
         return "https://rbi.org.in/Scripts/Publications.aspx?publication=HalfYearly"
    elif re.match(r"RBI Publications - Monthly \w+ \d{4}", ref):
         return "https://rbi.org.in/Scripts/Publications.aspx?publication=Monthly"
    elif 'Survey of Professional Forecasters on Macroeconomic Indicators' in ref:
        return "https://rbi.org.in/Scripts/Publications.aspx?publication=BiMonthly"
    elif re.match(r"RBI Publications Biennial \w+ \d{4}", ref):
        return "https://rbi.org.in/Scripts/Publications.aspx?publication=Biennial"
    elif re.match(r"Sources of Variation in India’s Foreign Exchange Reserves RBI Publications - Quaterly \w+ \w+ \d{4}", ref):
        return "https://rbi.org.in/Scripts/Publications.aspx?publication=Quarterly"
    elif re.match(r".+ - RBI Notifications \w+ \d{1,2}, \d{4}", ref):
        return "https://rbi.org.in/Scripts/NotificationUser.aspx"
    elif re.match(r"RBI - Occasional Papers - Vol\. \d{2}, No\. ?\d(?:,|:)? ?[A-Za-z]+ \d{1,2}, \d{4}", ref):
        return "https://rbi.org.in/Scripts/HalfYearlyPublications.aspx?head=Occasional+Papers"
    elif re.match(r"RBI WPS \(DEPR\): \d{2}/\d{4}: .+", ref):
        return "https://rbi.org.in/Scripts/PublicationsView.aspx?head=Working%20Papers"
    elif re.match(r"Measuring Productivity at the Industry Level – The India KLEMS Database \w+ \d{1,2}, \d{4}", ref):
        return "https://rbi.org.in/Scripts/KLEMS.aspx"
    elif re.match(r"RBI Publications - Weekly \d{1,2} \w+ \d{4}", ref):
        return "https://rbi.org.in/Scripts/Publications.aspx?publication=Weekly"
    elif re.match(r"RBI Publications - Reports .+ \d{1,2} \w+ \d{4}", ref):
        return "https://rbi.org.in/Scripts/Publications.aspx?publication=Reports"
    elif re.match(r"RBI Speeches - .+", ref):
        return "https://rbi.org.in/Scripts/BS_ViewSpeeches.aspx"
    elif re.match(r'DRG Study No\. \d{1,3}: .+ \w+ \d{1,2}, \d{4}', ref):
        return "https://rbi.org.in/Scripts/Occas_DRG_Studies.aspx"
    elif re.match(r".+ Press Release \w+ \d{1,2}, \d{4}", ref):
        return "https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
    elif re.match(r"Lending and Deposit Rates of Scheduled Commercial Banks – \w+ \d{4}", ref):
        return "https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
    elif re.match(r"Monthly Data on India’s International Trade in Services.+", ref):
        return "https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
    elif re.match(r"Scheduled Banks’ Statement of Position in India as on .+", ref):
        return "https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
    elif re.match(r"Sectoral Deployment of Bank Credit – \w+ \d{4}", ref):
        return "https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
    elif re.match(r"(THE\s+)?[A-Z][A-Za-z ’()\-]+(Act|Code), \d{4}", ref) or re.match(r".+Act, \d{4}", ref):
        return "https://rbi.org.in/Scripts/Act.aspx"
    elif re.match(r'.*Scheme, \d{4}$', ref):
        return 'https://rbi.org.in/Scripts/Schemes.aspx'
    elif re.match(r'.*Regulations, \d{4}$', ref):
        return 'https://rbi.org.in/Scripts/Regulations.aspx'
    elif re.match(r'.*Rules, \d{4}$', ref):
        return 'https://rbi.org.in/Scripts/Rules.aspx'
    # RBI Governor Speeches / Interviews / Press Conferences / Fireside Chats
    elif re.match(r"(Edited\s+)?Transcript of the Reserve Bank of India’s Post-Monetary Policy Press Conference: \w+ \d{1,2}, \d{4}", ref, re.IGNORECASE):
        return "https://rbi.org.in/Scripts/BS_SpeechesView.aspx"

    elif re.match(r"Edited transcript of Reserve Bank of India’s Governor Press Conference with Media: \w+ \d{1,2}, \d{4}", ref, re.IGNORECASE):
        return "https://rbi.org.in/Scripts/BS_SpeechesView.aspx"

    elif re.match(r"(Fireside chat|Panel Discussion) with Governor.*on \w+ \d{1,2}, \d{4}", ref, re.IGNORECASE):
        return "https://rbi.org.in/Scripts/BS_SpeechesView.aspx"

    elif re.match(r"Interview of Governor.*on \w+ \d{1,2}, \d{4}", ref, re.IGNORECASE):
        return "https://rbi.org.in/Scripts/BS_SpeechesView.aspx"

    elif re.match(r"Master Direction(s)?( –| -)? .+", ref, re.IGNORECASE):
        return "https://rbi.org.in/Scripts/BS_ViewMasterDirections.aspx"

    elif re.match(r".*(Draft|draft|DRAFT).*(Circular|Direction|Guideline|Framework|Regulation|Instruction).* \w+ \d{1,2}, \d{4}", ref):
        return "https://rbi.org.in/Scripts/DraftNotificationsGuildelines.aspx"
    elif re.match(r"Master Circular(s)?( –|-)? (on )?.+ \w+ \d{1,2}, \d{4}", ref, re.IGNORECASE):
        return "https://rbi.org.in/Scripts/BS_ViewMasterCirculardetails.aspx"
    elif re.match(r".+ -? ?PIB \d{1,2} \w+ \d{4}", ref, re.IGNORECASE):
        return "https://pib.gov.in/PressReleseDetail.aspx?PRID=2089308&reg=3&lang=1"
    elif re.match(r"MSME ANNUAL REPORT( \d{4}-\d{2})?$", ref, re.IGNORECASE):
        return "https://www.msme.gov.in/relatedlinks/annual-report-ministry-micro-small-and-medium-enterprises"
    elif re.match(r"Ministry Wise Procurement \d{4}-\d{2}", ref):
        return "https://sambandh.msme.gov.in/MinistryWisesReport.aspx"
    elif re.match(r"RBI Report On Trend And Progress Of Banking In India \d{4}-\d{2}", ref):
        return "https://www.rbi.org.in/Scripts/AnnualPublications.aspx?head=Trend+and+Progress+of+Banking+in+India"
    elif re.match(r"\d+-Year GST Statistical Report", ref, re.IGNORECASE):
        return "https://tutorial.gst.gov.in/offlineutilities/gst_statistics/6YearReport.pdf"
    elif re.match(r"India Budget \d{4}-\d{4}", ref, re.IGNORECASE):
        return "https://www.indiabudget.gov.in/doc/Budget_Speech.pdf"
    elif re.match(r"Udyog Aadhar Registeration \d{4}-\d{4}", ref, re.IGNORECASE):
        return "https://www.dcmsme.gov.in/uampublication.aspx"
    elif re.match(r"Udyog Aadhar Registeration \w+ \d{4}", ref, re.IGNORECASE):
        return "https://www.dcmsme.gov.in/uampublication.aspx"
    elif re.match(r"MALAYSIA DEVELOPMENT EXPERIENCE SME \w+ \d{4}", ref, re.IGNORECASE):
        return "https://documents1.worldbank.org/curated/en/504361583989615623/pdf/Malaysia-s-Experience-with-the-Small-and-Medium-Sized-Enterprises-Masterplan-Lessons-Learned.pdf"
    elif re.match(r"Malaysian SME Program Efficiency Review \w+ \d{4}", ref, re.IGNORECASE):
        return "https://documents1.worldbank.org/curated/en/099255003152238688/pdf/P17014606709a70f50856d0799328fb7040.pdf"
    else:
        return "Unknown Url"  # Default empty if no match

def synthesize_with_gemini(
    question: str,
    unstructured_results: List[Dict]
) -> str:
    """
    Synthesizes a final answer using Gemini-2.0-Flash based on unstructured sources.
    """

    # Format top results
    formatted_sources = ""
    for idx, result in enumerate(unstructured_results, start=1):
        content = result.get("content", "").strip()
        reference = result.get("reference", "").strip()
        url = result.get("url", "").strip()

        formatted_sources += (
            f"### Source {idx}:\n"
            f"- **Reference**: {reference}\n"
            f"- **URL**: {url if url else 'N/A'}\n"
            f"- **Extracted Content**:\n{content}\n\n"
        )

    # System instruction prompt without structured data logic
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""Based on the original question: {question}, and the following unstructured text data from various sources, synthesize a comprehensive and coherent answer. Integrate the information smoothly.

        **Formatting Instructions:**
        - Begin the final answer with this header: `## Insights from Ingested Data`
        - **When using any content from the 'Unstructured Text Data', cite the corresponding 'Reference' name and URL, but only if it is actually used.**
        - Indicate citations inline using square brackets like this: [1], [2], etc.
        - At the end of the answer, add a section titled `## References` that lists only the used references, numbered to match the inline citations.
        - Only include references that were cited in the text.
        - If no URL is available, skip the reference entirely.
        - Format the `## References` section exactly like this:

        ## References
        1. [Reference Name 1](https://example.com)
        2. [Reference Name 2](https://example.com)

        - Do not include any references that were not cited in the synthesized answer.
        - Avoid duplicate citations for the same source in the same paragraph — cite once per distinct point.
        - If data is conflicting or ambiguous, acknowledge that transparently in the summary.

            """),
            temperature=0.0,
            ),
        contents=formatted_sources
    )
    return response.text


# Search API Endpoint
@app.post("/search-topN", dependencies=[Depends(verify_api_key)])
async def search_topN_milvus(request: Request, question: Question):
    start_time = time.time()
    request_time = datetime.utcnow().isoformat()

    llm_query = clarify_query(question.question).strip()

    try:
        query_date = fetch_date(llm_query).strip()
        query_min_date = fetch_min_date(llm_query).strip()
        query_duration = abs(months_since(query_min_date,query_date))
        window_size = max(24,6 + min(18, query_duration))
        logging.info(f"Query min date: {query_min_date}, max date: {query_date}, Query duration is {query_duration}")
    except:
        query_date = 'today'
        window_size = 24
    months_after = int(max(1,min(window_size,2)))
    months_before = max(1,window_size - months_after)
    milvus_date_filter = build_range_around_date(query_date, months_before, months_after)["filter"]

    client_ip = request.client.host  # Get client IP address

    logging.info(f"Received request from {client_ip} at {request_time}")
    logging.info(f"Question Asked: {question.question}")
    logging.info(f"LLM Query Generated: {llm_query}")

    try:
        # Track token count
        token_count = len(question.question.split())  # Approximate token count
        logging.info(f"Question Token Count: {token_count}")

        llm_token_count = len(llm_query.split())  # Approximate token count
        logging.info(f"LLM Query Token Count: {llm_token_count}")

        # Start embedding generation
        embed_start = time.time()
        query_vector = emb_text(model, llm_query)#; logging.info(query_vector)
        embed_time = time.time() - embed_start

        logging.info(f"Embedding generation time: {embed_time:.4f} seconds")

        # Search in Milvus
        search_start = time.time()
        search_res = get_search_results(
            milvus_client, CPI_V5_COLLECTION_NAME, query_vector, ["content", "source", "id", "page", "reference", "date"],
            milvus_date_filter
        )
        search_time = time.time() - search_start
        logging.info(f"Milvus search execution time: {search_time:.4f} seconds")
        logging.info(f"Document search date filter: {milvus_date_filter}")

        if not search_res or not search_res[0]:
            logging.warning("No results found for query")
            raise HTTPException(status_code=404, detail="No results found")

        # Retrieve the top results
        top_15 = [
            {
                "content": result["entity"]["content"],
                "distance": result["distance"],
                "source": result["entity"]["source"],
                "page": result["entity"]["page"],
                "reference": result["entity"]["reference"],
                "date": result["entity"]["date"]
            }
            for result in search_res[0]
        ]

        # Log Top 15
        logging.info("Top 100 sources before reranking:")
        for i, item in enumerate(top_15, start=1):
            logging.info(
        #        f"Result - Content: {item['content']}, Page: {item['page']}, "
                f"Source: {item['source']}, Reference: {item['reference']}, Date: {item['date']}, Distance: {item['distance']:.4f}"
            )

        #  Rerank with CrossEncoder
        pairs = [(llm_query, item["content"]) for item in top_15]
        scores = cross_encoder.predict(pairs)
        # Let's assume each item in top_15 has a "date" field
        #date_boosts = [0.5 * months_since(item["date"],query_date) for item in top_15]
        deltas   = [(months_since(item["date"],query_date)) for item in top_15] # Signed deltas, positive = older and negative = newer than query date
        if min(deltas) > 0:
            # Date is too recent, we do not have matching documents
            maxdelta = min(deltas)
        else:
            # We have at least one document matching the query date
            if max(deltas) < 0:
                # Date is too old, we do not have documents that old
                maxdelta = max(deltas)
            else:
                maxdelta = 0
        maxdelta += 0.5*window_size
        mindelta = maxdelta - window_size
        chunks_found = False
        chunk_attempt = 0
        top_5_final = []
        while ((not chunks_found) and (chunk_attempt < 2) and (len(top_5_final) < 3)):
            chunk_attempt += 1
            logging.info("Deltas being used: " + str([mindelta,maxdelta]))
            #logging.info("Computed deltas")
            #logging.info(deltas)
            date_boosts = [0 if maxdelta >= deltas[xx] >= mindelta else 25 for xx in range(len(deltas))]

            # Add boosted scores to the cross_encoder scores
            #logging.info("Original scores: " + str(scores))
            final_scores = [s - boost for s, boost in zip(scores, date_boosts)]
            #logging.info("Modified scores: " + str(final_scores))

            # Now rerank based on the final boosted score
            reranked = sorted(
            zip(top_15, final_scores),
            key=lambda x: x[1],
            reverse=True
            )

            # Top 5 with cross_score filtering
            top_5_final = []
            content_concat = []
            cross_thresh = 3.0
            best_relevance = cross_thresh
            for item, score in reranked[:6]:
                item["cross_score"] = float(score)
                if (item["cross_score"] > cross_thresh) and (item["cross_score"] >= 0.9*best_relevance):  # Only include results where cross_score > threshold
                    # Attach the reference URL
                    item["url"] = get_reference_url(item["reference"])
                    top_5_final.append(item)
                    content_concat += item["content"]
                    #best_relevance = max(best_relevance,item["cross_score"])

            if not top_5_final:
                logging.warning("Not enough valid results found in current attempt: (" + str(len(top_5_final)) + "/5). Relaxing cross score and deltas ..")
                mindelta += 6
                maxdelta += 6
            else:
                chunks_found = True

        # Check if no valid results with cross_score > 0 were found

        if not top_5_final:
            logging.warning("No valid results with cross_score > 0")
            total_time = time.time() - start_time
            logging.info(f"Total processing time: {total_time:.4f} seconds")
            return {
                "question": question.question,
                "llm_query": llm_query,
                "query_date": query_date,
                "retrieved_results": [{
                    "content": "We could not find any relevant content related to your query.",
                    "distance": "N/A",
                    "source": "N/A",
                    "page": "N/A",
                    "reference": "N/A",
                    "date": "N/A",
                    "url": "N/A"
                }],
                "time": total_time,
            }
        else:
            # Log Top 5
            logging.info("Top 5 results after reranking:")
            for i, res in enumerate(top_5_final, start=1):
                logging.info(
                    f"{i}. Content: {res['content'][:200]}..., Page: {res['page']}, "
                    f"Source: {res['source']}, Reference: {res['reference']}, Date: {item['date']}, Distance: {res['distance']:.4f}, Cross Score: {res['cross_score']:.4f}"
                )
            total_time = time.time() - start_time
            logging.info(f"Total processing time: {total_time:.4f} seconds")
            return {
                "question": question.question,
                "llm_query": llm_query,
                "query_date": query_date,
                "retrieved_results": top_5_final,
                "time": total_time,
            }


    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        logging.error(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail=error_message)