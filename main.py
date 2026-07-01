from operator import index

from PyPDF2 import PdfReader
from dotenv import load_dotenv,dotenv_values
import json
from openai import AzureOpenAI
import os
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient, SearchIndexingBufferedSender 
from azure.search.documents.indexes import SearchIndexClient 
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryCaptionResult,
    QueryAnswerResult,
    SemanticErrorMode,
    SemanticErrorReason,
    SemanticSearchResultsType,
    QueryType,
    VectorizedQuery,
    VectorQuery,
    VectorFilterMode,    
)
from azure.search.documents.indexes.models import (  
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SearchFieldDataType,
    SearchIndex,  
    SearchField,   
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticPrioritizedFields,
    SemanticField,  
    SearchField,  
    SemanticSearch,
    VectorSearch,  
    HnswAlgorithmConfiguration,
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
    SearchIndex,
    SearchField,
    SimpleField,
    SearchableField,
    VectorSearch,
    ExhaustiveKnnParameters,
    SearchIndex,  
    SearchField,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticField,  
    SearchField,  
    VectorSearch,  
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)  

FILE_PATH = "ncert-science.pdf"
load_dotenv()
values_env = dotenv_values(".env")
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT") 
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME") 
key = os.getenv("AZURE_SEARCH_ADMIN_KEY") 
model = os.getenv("MODEL_NAME")
credential = AzureKeyCredential(key)



def get_pdf_data(file_path, num_pages = 1):
    reader = PdfReader(file_path)
    full_doc_text = ""
    pages = reader.pages
    num_pages = len(pages) 
    
    try:
        for page in range(num_pages):
            current_page = reader.pages[page]
            text = current_page.extract_text()
            full_doc_text += text
    except:
        print("Error reading file")
    finally:
        return full_doc_text
    
def get_chunks(fulltext:str,chunk_length =500) -> list:
    text = fulltext

    chunks = []
    while len(text) > chunk_length:
        last_period_index = text[:chunk_length].rfind('.')
        if last_period_index == -1:
            last_period_index = chunk_length
        chunks.append(text[:last_period_index])
        text = text[last_period_index+1:]
    chunks.append(text)

    return chunks

filename = FILE_PATH



def generate_embeddings(text, model,client):
    return client.embeddings.create(input = [text], model=model).data[0].embedding



def embed_document(filename:str, model:str,client:AzureOpenAI):
    
    full_doc_text = get_pdf_data(filename)
    Lines =get_chunks(full_doc_text,500)

    input_data = []
    counter = 1

    for line in Lines:
            d = {}
            d['id'] = str(counter)
            d['line'] = line
            d['embedding'] = generate_embeddings(line,model,client)
            d['filename'] = FILE_PATH.split('/')[-1]
            counter = counter + 1
            input_data.append(d)

    with open("docVectors.json", "w") as f:
          json.dump(input_data, f)


def upload_document_to_azure_search(index_name:str, service_endpoint:str, key:str, input_data:list):
    index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)
    fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, 
                key=True, sortable=True, 
                filterable=True, facetable=True),
    SearchableField(name="line", type=SearchFieldDataType.String),
    SearchableField(name="filename", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
    SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=3072, 
                vector_search_profile_name="myHnswProfile")
]
    vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE
            )
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw",
        )
    ]
)
    semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        content_fields=[SemanticField(field_name="line")],
        keywords_fields=[SemanticField(field_name="filename")]
    )
)
    
    semantic_search = SemanticSearch(configurations=[semantic_config])
    index = SearchIndex(name=index_name, fields=fields,
                    vector_search=vector_search, 
                    semantic_search=semantic_search)
    result = index_client.create_or_update_index(index)
    
    with open('docVectors.json', 'r') as file:  
        documents = json.load(file)  
    print(f' {result.name} created')
    search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)
    result = search_client.upload_documents(documents)
    print(f"Uploaded {len(documents)} documents") 


def search_documents(index_name:str, service_endpoint:str, key:str, query:str, model:str,client:AzureOpenAI):
    search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)
    quer_vector = generate_embeddings(query, model, client)
    results = search_client.search(
        search_text=query,
        vector_queries=[VectorizedQuery(
            vector=quer_vector,
            fields="embedding",
           
        )],
        select=["line", "filename"],
        top=3,
    )
    results_list = list(results)
    for result in results_list:
        print(f"Score: {result['@search.score']}, Line: {result['line']}, Filename: {result['filename']}")
     
    return results_list

def main():

    model: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ID")
   
    client = AzureOpenAI(
                api_key = os.getenv("AZURE_OPENAI_KEY"),  
                api_version = "2023-05-15",
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                )

    question = "What is the main topic of the document?"
    result=search_documents(index_name, service_endpoint, key, question, model, client)
    
    context = "\n".join([doc["line"] for doc in result])

    generation_model = os.getenv("MODEL_NAME")

    response = client.chat.completions.create(
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant. Answer the question based only on the provided context. If the answer isn't in the context, say you don't know."
        },
        {
            "role": "user", 
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ],
    model=generation_model,
    temperature=0.7,
    max_tokens=1000
)

# Step 4: Print the answer
    print(response.choices[0].message.content)
 



if __name__ == "__main__":
    main()
