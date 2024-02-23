import hashlib
import os
import sys
import tempfile
from datetime import datetime

import dotenv
import nest_asyncio
import requests
import streamlit as st
from bs4 import BeautifulSoup
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


def get_soup(link):
    """
    Return the BeautifulSoup object for input link
    """
    request_object = requests.get(link, auth=("user", "pass"))
    soup = BeautifulSoup(request_object.content, "lxml")
    return soup


def get_status_code(link):
    """
    Return the error code for any url
    param: link
    """
    try:
        error_code = requests.get(link).status_code
    except requests.exceptions.ConnectionError:
        error_code = -1
    return error_code


def find_internal_urls(main_url, depth=0, max_depth=2):
    all_urls_info = []

    soup = get_soup(main_url)
    a_tags = soup.findAll(
        "a",
        {
            "class": [
                "sidebar-item-text sidebar-link active",
                "sidebar-item-text sidebar-link",
            ]
        },
        href=True,
    )

    if main_url.endswith("/"):
        domain = main_url
    else:
        domain = "/".join(main_url.split("/")[:-1])

    if depth > max_depth:
        return {}
    else:
        for a_tag in a_tags:
            if (
                "http://" not in a_tag["href"]
                and "https://" not in a_tag["href"]
                and "/" in a_tag["href"]
            ):

                url = domain.rstrip("docs/") + a_tag["href"].lstrip("..")

            elif "http://" in a_tag["href"] or "https://" in a_tag["href"]:
                url = a_tag["href"]
            else:
                continue

            # status_dict = {}
            # status_dict["url"] = url
            # status_dict["status_code"] = get_status_code(url)
            # status_dict["timestamp"] = datetime.now()
            # status_dict["depth"] = depth + 1
            # all_urls_info.append(status_dict)
            all_urls_info.append(url)

    return all_urls_info


@st.cache_resource
def configure_retriever(url):
    # Read documents
    # nest_asyncio.apply()
    # Load HTML
    loader = AsyncChromiumLoader(url)
    html = loader.load()
    print("loaded")
    # Transform
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        html, tags_to_extract=["p", "span"]
    )

    # Result
    # docs_transformed
    text = docs_transformed[0].page_content
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

# url = ["https://shiny.posit.co/py/docs/dashboards.html"]

url = [
    "https://shiny.posit.co/py/docs/quick-start.html",
    "https://shiny.posit.co/py/docs/dashboards.html",
    "https://shiny.posit.co/py/docs/jupyter-widgets.html",
    "https://shiny.posit.co/py/docs/install-create-run.html",
    "https://shiny.posit.co/py/docs/debug.html",
    "https://shiny.posit.co/py/docs/reactive-foundations.html",
    "https://shiny.posit.co/py/docs/reactive-patterns.html",
    "https://shiny.posit.co/py/docs/reactive-mutable.html",
    "https://shiny.posit.co/py/docs/ui-components.html",
    "https://shiny.posit.co/py/docs/ui-dynamic.html",
    "https://shiny.posit.co/py/docs/ui-html.html",
    "https://shiny.posit.co/py/docs/ui-customize.html",
    "https://shiny.posit.co/py/docs/express-introduction.html",
    "https://shiny.posit.co/py/docs/express-in-depth.html",
    "https://shiny.posit.co/py/docs/express-to-core.html",
    "https://shiny.posit.co/py/docs/modules.html",
    "https://shiny.posit.co/py/docs/module-communication.html",
    "https://shiny.posit.co/py/docs/custom-component-one-off.html",
    "https://shiny.posit.co/py/docs/custom-components-pkg.html",
    "https://shiny.posit.co/py/docs/comp-streamlit.html",
    "https://shiny.posit.co/py/docs/comp-r-shiny.html",
    "https://shiny.posit.co/py/docs/nonblocking.html",
    "https://shiny.posit.co/py/docs/routing.html",
]
# url_list = find_internal_urls(url)
# url_list
# nest_asyncio.apply()
# loader = AsyncChromiumLoader(
#     # [
#     #     "https://shiny.posit.co/py/docs/dashboards.html",
#     #     "https://shiny.posit.co/py/docs/quick-start.html"
#     # ]
#     url_list
# )
# html = loader.load()
# html

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
