import os

import pandas as pd
from dotenv import load_dotenv
from langchain.llms import OpenAI as Langchain_openai
from pandasai import SmartDataframe
from pandasai.helpers.openai_info import get_openai_callback
from pandasai.llm import OpenAI

# OpenAI models
# load_dotenv()
# APY_KEY = os.environ["OPENAI_API_KEY"]
# llm = OpenAI(api_token=APY_KEY)
# below set API as environmet variable, could instantiate the OpenAI object without pass the API key
# add temperature and seed to enhance determinism, avoid randomness of answer
llm = OpenAI(temperature=0, seed=26)
langchain_llm = Langchain_openai()

# df = pd.read_csv("data/hc_model_data.csv")
df = pd.read_excel("data/ms_sample.xlsx")
# df.info()


# convert to SmartDataframe
# conversational=False is supposed to display lower usage and cost
df_smart = SmartDataframe(
    df,
    config={
        "llm": llm,
        # "llm": langchain_llm,
        "conversational": False,
        "custom_whitelisted_dependencies": ["re", "collections"],
    },
)


with get_openai_callback() as cb:
    chat_content = (
        "For osat less than 5, what are commonly mentioned complaints in comments "
    )
    response = df_smart.chat(chat_content)
    print(response)
    print(cb)
