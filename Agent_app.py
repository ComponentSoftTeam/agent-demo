# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from operator import itemgetter
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# + active=""
# from langsmith import Client
#
# import os
# os.environ["LANGCHAIN_PROJECT"] = "Agents with Langchain"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# #os.environ["LANGCHAIN_TRACING_V2"] = "false"
#
# client = Client()
# -

import os

News_api_key = os.environ["NEWS_API_KEY"]
Financial_news_api_key = os.environ["ALPHAVANTAGE_API_KEY"]

# + active=""
# def check_var_in_env_file(var_name, env_file_path='.env'):
#     """Check if a variable exists in a .env file"""
#     var_exists = False
#     if os.path.exists(env_file_path):
#         with open(env_file_path, 'r') as f:
#             for line in f:
#                 if line.startswith(var_name):
#                     var_exists = True
#                     break
#     return var_exists

# + active=""
# if check_var_in_env_file('GCP'):
#     !gcloud auth application-default login
# -

from langchain_openai import ChatOpenAI

from langchain_google_genai import ChatGoogleGenerativeAI

# from local_google_genai_chat_models import ChatGoogleGenerativeAI

# from langchain_google_vertexai import ChatVertexAI
# from local_google_vertexai_chat_models import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_fireworks.chat_models import ChatFireworks
from langchain_core.output_parsers.string import StrOutputParser

# +
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

# from langchain.agents import create_tool_calling_agent, create_react_agent, AgentExecutor
from langchain.agents import create_tool_calling_agent, create_react_agent
from local_agent import AgentExecutor

# from langchain_community.agent_toolkits.load_tools import load_tools
from local_load_tools import load_tools
from langchain_core.prompts import ChatPromptTemplate
from langchain.globals import set_debug, set_verbose
from langchain_core.output_parsers.string import StrOutputParser
from langchain import hub

set_debug(False)
set_verbose(False)

# +
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 1000

LLM_MODELS = {
    "llama-3-70b-prompting": ChatFireworks(
        model_name="accounts/fireworks/models/llama-v3-70b-instruct",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
    "Llama-3-70b-firefunction-v2": ChatFireworks(
        model_name="accounts/fireworks/models/firefunction-v1",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
    "gpt-3.5-turbo": ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
    "gpt-4o": ChatOpenAI(
        model="gpt-4o",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
    "gpt-4-turbo": ChatOpenAI(
        model="gpt-4-turbo",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
    "mistral-large": ChatMistralAI(
        model="mistral-large-latest",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
    "open-mixtral-8x22b": ChatMistralAI(
        model="open-mixtral-8x22b",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
    "mistral-small": ChatMistralAI(
        model="mistral-small-latest",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
    "gemini-1.5-flash": ChatGoogleGenerativeAI(
        # "gemini-1.5-flash": ChatVertexAI(
        model="gemini-1.5-flash-latest",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
    "gemini-1.5-pro": ChatGoogleGenerativeAI(
        # "gemini-1.5-pro": ChatVertexAI(
        model="gemini-1.5-pro-latest",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
    "claude-3-haiku": ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
    "claude-3.5-sonnet": ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
    "claude-3-opus": ChatAnthropic(
        model="claude-3-opus-20240229",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    ),
}

model_type = "gemini-1.5-pro"
llm = LLM_MODELS[model_type]

parser = StrOutputParser()
# -

# +
DATE = datetime.today().strftime("%Y-%m-%d")

wikipedia_tool, arxiv_tool, websearch_tool, llm_math_tool, weather_tool, news_tool = (
    load_tools(
        ["wikipedia", "arxiv", "ddg-search", "llm-math", "open-meteo-api", "news-api"],
        news_api_key=News_api_key,
        llm=llm,
    )
)

# llm_math_tool.invoke("What is the square root of 4?")
# wikipedia_tool.invoke("How many people live in Prague?")
# print(weather_tool.invoke(prompt, verbose=False))
# print(news_tool.invoke("What are the 10 most important news in Prague? Answer in 10 bullet points"))
# print(arxiv_tool.invoke("List the title of 10 scientific papers about LLM agents published in this year.", verbose=True))
# print(websearch_tool.invoke("Who won the most Oscar in this year?"))
# print(pubmed_tool.invoke("List papers about Vortioxetin."))llm
# -


tools = [
    llm_math_tool,
    arxiv_tool,
    news_tool,
    weather_tool,
    wikipedia_tool,
    websearch_tool,
]  # More tools could be added

# +


if model_type.startswith("llama"):
    prompt_react = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt_react)
else:
    prompt_tool_calling = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt_tool_calling)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, stream_runnable=False
)

agent_chain = agent_executor | itemgetter("output")

# +
system_prompt = f"You're a helpful assistant. Always use tools to answer questions. Always use the Calculator for calculations, even when adding 2 numbers or calculating averages. The current date is {datetime.today().strftime('%Y-%m-%d')}"

# prompt = "What is the square root of 2?"
# prompt = "What is the weather like in Budapest?"
# prompt = "What are the 10 trending news in Budapest Hungary? Answer in 10 bullet points."
# prompt = "How many people live in Budapest?"
# prompt = "Who won the most Oscar in this year?"
# prompt = "List the title of 10 scientific papers about LLM agents published in this year."

# prompt = "What are the maximum, minimum and average temperature values as well as the sum of precipitations on each of the coming 3 days in Budapest?"
# prompt = f"Create a table that contains the date as well as the maximum, minimum and average temperature values as well as the sum of precipitations in the coming 7 days in Budapest"
prompt = f"""Create a table that contains the date as well as the maximum, minimum and average temperature values as well as the sum of precipitations separately for each of the coming 7 days in Budapest.
Then include a line with each of the average of the maximum, minimum and average temperature values for these 7 days. And after all that also write a list with 10 current news in Budapest Hungary."""

streaming = False

response = ""
if streaming == True:
    for new_token in agent_chain.stream(
        {"input": prompt}, {"callbacks": [VariableCallbackHandler(session_id)]}
    ):
        print(new_token, end="")
        response = response + new_token
else:
    response = agent_chain.invoke({"system_prompt": system_prompt, "input": prompt})
    print(response)
print("")
# -

'''
# agent_executor.invoke({"input": "How many people live in Budapest and Prague together?"})
print(
    agent_chain.invoke(
        {"system_prompt": system_prompt, "input": "What is the square root of 2?"}
    )
)

print(
    agent_chain.invoke(
        {
            "system_prompt": system_prompt,
            "input": "What is the weather like in Budapest?",
        }
    )
)

print(
    agent_chain.invoke(
        {
            "system_prompt": system_prompt,
            "input": "What are the 10 trending news in Budapest Hungary? Answer in 10 bullet points.",
        }
    )
)

print(
    agent_chain.invoke(
        {"system_prompt": system_prompt, "input": "How many people live in Budapest?"}
    )
)

print(
    agent_chain.invoke(
        {
            "system_prompt": system_prompt,
            "input": "Who won the most Oscar in this year according to a web search?",
        }
    )
)

print(
    agent_chain.invoke(
        {
            "system_prompt": system_prompt,
            "input": "List the title of 10 scientific papers about LLM agents published in this year.",
        }
    )
)

print(
    agent_chain.invoke(
        {
            "system_prompt": system_prompt,
            "input": "What are the maximum, minimum and average temperature values as well as the sum of precipitations on each of the coming 3 days in Budapest?",
        }
    )
)

print(
    agent_chain.invoke(
        {
            "system_prompt": system_prompt,
            "input": "Create a table that contains the date as well as the maximum, minimum and average temperature values as well as the sum of precipitations in the coming 7 days in Budapest?",
        }
    )
)

print(
    agent_chain.invoke(
        {
            "system_prompt": system_prompt,
            "input": """Create a table that contains the date as well as the maximum, minimum and average temperature values as well as the sum of precipitations separately for each of the coming 7 days in Budapest.
Then include a line with each of the average of the maximum, minimum and avarage temperature values for these 7 days. And after all that also write a list with 10 current news in Budapest Hungary.""",
        }
    )
)
'''
