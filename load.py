from langchain.vectorstores import Qdrant
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

import products

load_dotenv()

# we will use OpenAI as our embeddings provider
embedding = OpenAIEmbeddings()

# name of the Redis search index to create
collection_name = "products"

# create and load qdrant with documents
vectorstore = Qdrant.from_texts(
    texts=products.texts,
    # metadatas=products.metadatas,
    embedding=embedding,
    host="localhost",
    collection_name=collection_name,
)
