import os

import dotenv
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()
API_KEY = os.environ["OPENAI_API_KEY"]


# @st.cache_resource
# def get_script(url):
#     """
#     url: string, youtube video url
#     retun: string
#     """
#     loader = YoutubeLoader.from_youtube_url(
#         url,
#         add_video_info=True,
#     )
#     docs = loader.load()
#     print("loaded")
#     text = docs[0].page_content

#     return text


# @st.cache_resource
# def process_script(text):
#     """
#     return a vector db
#     """
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
#     splits = text_splitter.split_text(text)

#     # Build an index
#     embeddings = OpenAIEmbeddings()
#     vectordb = FAISS.from_texts(splits, embeddings)

#     return vectordb


# def generate_chain(vectordb):
#     # text = get_script(url)
#     # vectordb = process_script(text)
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3),
#         chain_type="stuff",
#         retriever=vectordb.as_retriever(),
#     )

#     return qa_chain


# def generate_response(chain, query):

#     answer = chain.run(query)
#     st.info(answer)


# # url = "https://www.youtube.com/watch?v=J4LtodhlOQ0&list=PLyzi0hs72I9rI8SQZCw8FE89aAelq9IOs&index=1"

# url = st.sidebar.text_input("YouTube url")
# text = get_script(url)
# vectordb = process_script(text)
# chain = generate_chain(vectordb)

# with st.form("my_form"):
#     text = st.text_area("Ask me anything", "")
#     submitted = st.form_submit_button("Submit")

#     if submitted and not url:
#         answer = generate_response(chain, text)


# # Ask a question!
# query = "What this video is talking about"
# query = "I know nothing about GA and attribution"
# query = "what are other models mentioned"
# query = "what is lask non-direct click?"


import os
import tempfile

import dotenv
import nest_asyncio
import streamlit as st
from langchain.agents import AgentExecutor, ConversationalChatAgent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@st.cache_resource
def configure_retriever(url):
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=True,
    )
    docs = loader.load()
    print("loaded")
    text = docs[0].page_content

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(text)

    # Build an index
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_texts(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
    )

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


st.set_page_config(page_title="Chat with Shiny for Python", page_icon="🦜")
st.title("🦜 LangChain: chat with everything")

dotenv.load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]

url = st.sidebar.text_input("YouTube url")
retriever = configure_retriever(url)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True
)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    # model_name=" gpt-4-0125-preview",
    openai_api_key=openai_api_key,
    temperature=0,
    streaming=True,
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(
            user_query, callbacks=[retrieval_handler, stream_handler]
        )
