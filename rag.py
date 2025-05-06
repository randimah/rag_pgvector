import datasets
import psycopg2
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

conn = psycopg2.connect("dbname=vectordb user=randimah")
cur = conn.cursor()

ollama_embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
)

llm = ChatOllama(model="llama3.2", temperature=0.7,)

def search_documents(query, limit=5):
    query_embedding = ollama_embeddings.embed_documents([query])[0]
    cur.execute("""
        SELECT content, source, embedding <-> %s AS distance
        FROM documents
        ORDER BY distance
        LIMIT %s
    """, (str(query_embedding), limit))
    return cur.fetchall()

# Perform a search
search_query = "What is the purpose of the Diffusion model?"
results = search_documents(search_query, limit=3)
print(f"Search results for: '{search_query}'")

page_contents = []
for i, (content, source, distance) in enumerate(results, 1):
    print(f"{i}. {source} (Distance: {distance:.4f})")
    page_contents.append(content)

# generate rag response
RAG_PROMPT_TEMPLATE = """You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """

docs_content = "\n\n".join(page_content for page_content in page_contents)
final_prompt = RAG_PROMPT_TEMPLATE.format(question=search_query,
                                          documents=docs_content)
response = llm.invoke(final_prompt)
print('Generated answer:', response.content)
