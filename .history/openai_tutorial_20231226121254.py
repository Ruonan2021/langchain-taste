import os

from dotenv import load_dotenv
from pandasai import 
from pandasai.llm.openai import OpenAI

load_dotenv()

API_KEY = os.environ["OPENAI_API_KEY"]
print(API_KEY)
# client = OpenAI()

# completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
#         },
#         {
#             "role": "user",
#             "content": "Compose a poem that explains the concept of recursion in programming.",
#         },
#     ],
# )

# print(completion.choices[0].message)
