import os

import bs4
import dotenv
from langchain.document_loaders import WebBaseLoader

dotenv.load_dotenv()
API_KEY = os.environ["OPENAI_API_KEY"]
API_KEY
# # inspect with LangSmith
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

# 1. Load data
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    },
)
docs = loader.load()
len(docs[0].page_content)
print(docs[0].page_content[:500])
print(type(docs[0]))  # <class 'langchain_core.documents.base.Document'>


# 2. Split data
# loaded document is over 42k characters long. This is too long to fit in the context window of many models.
# we’ll split the Document into chunks for embedding and vector storage
from langchain.text_splitter import RecursiveCharacterTextSplitter

# question here: how to determine chunk size and chunk overlap
# answer: https://python.langchain.com/docs/modules/data_connection/document_transformers/
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
# we got 66 chunks iin total
len(all_splits)
len(all_splits[0].page_content)
all_splits[65].metadata
len(all_splits[65].page_content)
# question here: start_index of all_splits[65] is 42174, len of page_content is 649, 42174+649 is not the len of len(docs[0].page_content) which is 42824

all_splits[0].metadata
len(all_splits[0].page_content)
all_splits[1].metadata
# question here: why 8+969 is not the start_index of all_splits[1]

# 3. Store data
# embed the contents of each document split and upload those embeddings to a vector store
# cosine sumularity
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=OpenAIEmbeddings(api_key=API_KEY)
)

# 4. Retrive data
# other retriever method: https://python.langchain.com/docs/modules/data_connection/retrievers/
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.get_relevant_documents(
    "What are the approaches to Task Decomposition?"
)
print(retrieved_docs[0].page_content)

# 5. Generate
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

from langchain import hub

# LangChain prompt hub
# question here
prompt = hub.pull("rlm/rag-prompt")
print(
    prompt.invoke(
        {"context": "filler context", "question": "filler question"}
    ).to_string()
)


from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# question here, to learn details of chain design
# TODO: check
# 1. prompt could be customised
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  # retriever
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)
