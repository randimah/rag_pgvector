# RAG with PGVector and Langchain
A RAG implementation with PGVector and Langchain.

## Prerequisites 
* Postgres SQL server should be installed and running locally
* Compile and install the pgvector extension https://github.com/pgvector/pgvector

### Setting up the Vector Database and Document Table
* Go to terminal and enter `psql` - this will initiate PostgreSQL 
  interactive terminal
* You could create a user for the new vector database using `CREATE USER 
xxx WITH PASSWORD xxx;`
* Create the new vector database with `CREATE DATABSE vectordb;`
* Select the newly created vector db with `\c vectordb;` (all available 
  databases can be listed with `\l`)
* Create the PGVector extension for the selected database with `CREATE 
EXTENSION vector;`
* Create `documents` table in the vector database. Remember to set the 
  vector dimension to match with your embedding model. 
```commandline
CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
		source TEXT, 
        embedding vector(1024))
```
* Create an index to accelerate search (Optional but highly recommended). 
  PGVector supports two 
  types of indexes.
    * *IVFFlat (Inverted File Flat) index*: Use IVFFlat when you need exact 
      results and can tolerate slightly slower searches
    * *HNSW (Hierarchical Navigable Small World) index*: Use HNSW when you  
      need fast searches and can accept slight inaccuracies
```commandline
CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```
```commandline
CREATE INDEX ON documents USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);
```
```commandline
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```
## Setting up the project
* Once the project is cloned, create a virtual environment and install 
  requirements.txt `pip install -r requirements.txt`
* 



