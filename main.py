from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant

load_dotenv()

# we will use OpenAI as our embeddings provider
embedding = OpenAIEmbeddings()


template = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.
Or end the conversation if it seems like it's done.
Chat History:\"""
{chat_history}
\"""
Follow Up Input: \"""
{question}
\"""
Standalone question:"""

condense_question_prompt = PromptTemplate.from_template(template)

template = """You are a friendly, conversational retail shopping assistant. Use the following context including product names, descriptions, and keywords to show the shopper whats available, help find what they want, and answer any questions.
 
It's ok if you don't know the answer.
Context:\"""
 
{context}
\"""
Question:\"
\"""
 
Helpful Answer:"""

qa_prompt = PromptTemplate.from_template(template)

# define two LLM models from OpenAI
llm = OpenAI(temperature=0)

streaming_llm = OpenAI(
    streaming=True,
    callback_manager=CallbackManager([
        StreamingStdOutCallbackHandler()
    ]),
    verbose=True,
    max_tokens=150,
    temperature=0.2
)

# use the LLM Chain to create a question creation chain
question_generator = LLMChain(
    llm=llm,
    prompt=condense_question_prompt
)

# use the streaming LLM to create a question answering chain
doc_chain = load_qa_chain(
    llm=streaming_llm,
    chain_type="stuff",
    prompt=qa_prompt
)

# create and load qdrant with documents
client = QdrantClient(host="localhost")
collection_name = "products"
vectorstore = Qdrant(client, collection_name,
                     embedding_function=embedding.embed_query)

chatbot = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(),
    combine_docs_chain=doc_chain,
    question_generator=question_generator
)

# create a chat history buffer
chat_history = []
# gather user input for the first question to kick off the bot
question = input("Hi! What are you looking for today?")

# keep the bot running in a loop to simulate a conversation
while True:
    result = chatbot(
        {"question": question, "chat_history": chat_history}
    )
    print("\n")
    chat_history.append((result["question"], result["answer"]))
    question = input()
