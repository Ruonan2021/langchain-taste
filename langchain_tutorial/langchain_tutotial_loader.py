import os

import dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

dotenv.load_dotenv()
API_KEY = os.environ["OPENAI_API_KEY"]


# Document laoders is one of the components
loader = CSVLoader(file_path="data/ms_sample.csv")
data = loader.load()

data

import re


def extraction_from_page_content(pattern, input_string):
    match = re.search(pattern, input_string)
    if match:
        # Extract the comments from the match
        search_result = match.group(1)
    return search_result


less_thann_5 = ""
greater_than_5 = ""
for item in data:
    item_content = item.page_content
    pattern_osat = r"Q1_OSAT:\s*(\d+)(?=\nComments)"
    pattern_comment = r"Comments:\s*(.*)$"

    q1_osat_value = int(extraction_from_page_content(pattern_osat, item_content))
    comment_value = extraction_from_page_content(pattern_comment, item_content)
    # print(q1_osat_value)
    if q1_osat_value <= 5:
        less_than_5 = "\n\n".join([less_than_5, comment_value])
    else:
        greater_than_5 = "\n\n".join([greater_than_5, comment_value])


less_than_5
greater_than_5


# text splitters
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([less_than_5])

# embeddings and vectorstore
vectorstore = Chroma.from_documents(texts, OpenAIEmbeddings(openai_api_key=API_KEY))
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retriever = vectorstore.as_retriever()


model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=model, memory_key="chat_history", return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever, memory=memory)

question = "why members are unsatisfied"
question = "Could you provide more details"
question = "Could you provide related comments count related to each issue?"
question = "Could you summarise top 10 issues mentioned in the comments"
question = "how many mentioned long wait"
question = "what are they"
qa(question)


# csv agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent

agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    "data/ms_sample.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

agent.run("how many rows are there?")
agent.run("how many rows for osat less than 5")
agent.run("summarise comments for osat less than 5")


template = """ Answer the question based only on the following context:{context}

uation: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


chain.invoke("How many record in the data")
