Azure AI Search RAG

A Retrieval-Augmented Generation (RAG) project that combines Azure AI Search (vector + semantic search) with Azure OpenAI chat completions to answer questions grounded in your own documents.

How It Works


Documents are indexed in Azure AI Search with both text fields and vector embeddings.
A user question is embedded and used to run a hybrid search (vector + keyword) with semantic ranking enabled.
The top matching results are assembled into a context string.
The context and question are sent to an Azure OpenAI chat model, which answers strictly based on the retrieved context.


Features


Vector search using HNSW and Exhaustive KNN algorithm configurations
Semantic ranking via a configured semantic configuration (content + keyword fields)
Retrieval-Augmented Generation using Azure OpenAI chat completions


Prerequisites


Python 3.9+
An Azure AI Search resource (with an index already created)
An Azure OpenAI resource with a deployed chat model and an embedding model


Setup


Clone the repo:


bash   git clone https://github.com/georgenevin/azure-ai-search-rag.git
   cd azure-ai-search-rag


Install dependencies:


bash   pip install -r requirements.txt


Create a .env file in the project root with your credentials:


   AZURE_SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
   AZURE_SEARCH_KEY=<your-search-admin-key>
   AZURE_SEARCH_INDEX=<your-index-name>
   AZURE_OPENAI_ENDPOINT=https://<your-openai-resource>.openai.azure.com
   AZURE_OPENAI_KEY=<your-openai-key>
   AZURE_OPENAI_CHAT_DEPLOYMENT=<your-chat-model-deployment-name>
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<your-embedding-model-deployment-name>


⚠️ Never commit your .env file. It is already excluded via .gitignore.



Usage

Run the main script and ask a question:

bashpython main.py

The script will:


Embed your question
Search the Azure AI Search index for relevant content
Send the retrieved context + question to the chat model
Print the grounded answer


Project Structure

AzureAISearch/
├── main.py           # Entry point: search + RAG chat completion
├── requirements.txt  # Python dependencies
├── .gitignore
└── README.md

Notes


The vector search uses VectorizedQuery with a pre-computed embedding vector.
Semantic search requires a semantic_configuration_name matching the configuration defined on your index.
Search results are materialized into a list (not left as a single-use iterator) so they can be reused when building context.


License

MIT