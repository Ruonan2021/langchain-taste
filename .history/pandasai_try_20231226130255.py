import os

import pandas as pd
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

# pandas dataframe
# df = pd.DataFrame(
#     {
#         "country": [
#             "United States",
#             "United Kingdom",
#             "France",
#             "Germany",
#             "Italy",
#             "Spain",
#             "Canada",
#             "Australia",
#             "Japan",
#             "China",
#         ],
#         "gdp": [
#             19294482071552,
#             2891615567872,
#             2411255037952,
#             3435817336832,
#             1745433788416,
#             1181205135360,
#             1607402389504,
#             1490967855104,
#             4380756541440,
#             14631844184064,
#         ],
#         "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12],
#     }
# )

df = pd.read_csv("data/hc_model_data.csv")

load_dotenv()
APY_KEY = os.environ["OPENAI_API_KEY"]
llm = OpenAI(api_token=APY_KEY)

# convert to SmartDataframe
df = SmartDataframe(df, config={"llm": llm})

response = df.chat("Calculate the sum of the gdp of north american countries")

df.chat("Describe how spend data varied overtimes")
print(response)
# Output: 20901884461056
