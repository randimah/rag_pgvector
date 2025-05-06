import psycopg2
from langchain_ollama import OllamaEmbeddings
from data_loader import load_data
from data_processor import split_documents

"""

This module loads the document dataset, split into chunk, vectorize text 
chunks and insert into documents table in the vector DB.

Note: `mxbai-embed-large` is used as the embedding model. There's a dimension 
mismatch between the models in ollama (1024) and hugging face (512). HF 
model is only used to tokenize text in order to chunk the document text.

"""
# embedding model
OLLAMA_EMBEDDING_MODEL_NAME = "mxbai-embed-large"
HF_EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
ollama_embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL_NAME, )

# Connect to the database
conn = psycopg2.connect("dbname=VECTOR_DBNAME user=USER")
cur = conn.cursor()

def add_document(content, source):
    embedding = ollama_embeddings.embed_documents([content])
    cur.execute(
        "INSERT INTO documents (content, embedding, source) VALUES (%s, %s, "
        "%s)",
        (content, embedding[0], source))
    conn.commit()

# load the dataset
knowledge_base = load_data()
# process documents
docs_processed = split_documents(
    512,  # We choose a chunk size adapted to the model
    knowledge_base,
    tokenizer_name=HF_EMBEDDING_MODEL_NAME,
)

# add documents to the vector db
for doc in docs_processed:
    add_document(doc.page_content, doc.metadata['source'])

cur.close()
conn.close()