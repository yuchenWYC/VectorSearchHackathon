import streamlit as st
from AtlasClient import AtlasClient
from OpenAIClient import OpenAIClient
from dotenv import find_dotenv, dotenv_values
from urllib.request import urlopen
import time
import openai
import pymongo

DB_NAME = 'PromptingPractice'
COLLECTION_NAME = 'practices'
INDEX_NAME = 'vector_index'


## Initialize only once
@st.cache_resource
def initialize():
    config = dotenv_values(find_dotenv())

    ATLAS_URI = config.get('ATLAS_URI')
    OPENAI_API_KEY = config.get("OPENAI_API_KEY")

    if not ATLAS_URI:
        raise Exception ("'ATLAS_URI' is not set.  Please set it above to continue...")

    if not OPENAI_API_KEY:
        raise Exception ("'OPENAI_API_KEY' is not set.  Please set it above to continue...")

    ip = urlopen('https://api.ipify.org').read()
    print (f"My public IP is '{ip}.   Make sure this IP is allowed to connect to cloud Atlas")

    atlas_client = AtlasClient (ATLAS_URI, DB_NAME)
    print ('Atlas client initialized')

    openAI_client = OpenAIClient (api_key=OPENAI_API_KEY)
    print ("openAI client initialized")

    return atlas_client, openAI_client
## ---------------------------

## ---------------------------
def run_query (query, output_container):
    config = dotenv_values(find_dotenv())
    ATLAS_URI = config.get('ATLAS_URI')
    OPENAI_API_KEY = config.get("OPENAI_API_KEY")
    client = pymongo.MongoClient(ATLAS_URI)
    db = client.PromptingPractice
    collection = db.practices
    model = "text-embedding-3-small"
    def generate_embedding(text: str) -> list[float]:
        openai.api_key = OPENAI_API_KEY
        return openai.embeddings.create(input = [text], model=model).data[0].embedding

    output_container.empty()
    with output_container.container():
        st.write (f'running query : {query}')

        t2a = time.perf_counter()
        # results = atlas_client.vector_search(query, collection_name=COLLECTION_NAME, index_name=INDEX_NAME, attr_name='plot_embedding', limit=10)
        results = collection.aggregate([
    {
        '$vectorSearch': {
            "index": INDEX_NAME,
            "path": 'description_embedding',
            "queryVector": generate_embedding(query),
            "numCandidates": 50,
            "limit": 5,
        }
    },
    {
          "$project": {
              '_id' : 1,
              'ex_problem' : 1,
              "search_score": { "$meta": "vectorSearchScore" }
        }
    }
    ])
        t2b = time.perf_counter()
        st.write (f"Altas query returned results in {(t2b-t2a)*1000:,.0f} ms")
        for idx, result in enumerate (results):
            res_str =  (f"""
- Practice {idx+1}
  - search score: {result["search_score"]}
  - id: {result["_id"]}
  - Problem: {result["ex_problem"]}
""")
            st.markdown(res_str)
## ---------------------------

## init, and do it only once
atlas_client, openAI_client = initialize()

## Process queries
st.header("Transcend Prompt Engineering Vector Search Portal📈")
st.markdown("""Here are some sample queries you can try...
- Young professional who uses Excel a lot
- SpongeBob is young and known for his extremely expressive and enthusiastic personality
""")
query_text = st.text_input("Practice problem query:")
output_container = st.empty()

if query_text:
    run_query (query_text, output_container)
