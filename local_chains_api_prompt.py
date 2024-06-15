# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

from datetime import datetime ###
DATE = datetime.today().strftime('%Y-%m-%d') ###

API_URL_PROMPT_TEMPLATE = f"The current date is {DATE}. " + """You are given the below API Documentation:
{api_docs}

Using this documentation, generate the full API url to call for answering the user question.
You should build the API url in order to get a response that is as short as possible, while still getting the necessary information to answer the question. Pay attention to deliberately exclude any unnecessary pieces of data in the API call.

Don't include apiKEY in the full API url, it's not needed. 
DON'T BE TALKATIVE JUST GENERATE THE FULL API URL WITHOUT ANY ADDITIONAL TEXT.

Question: {question}
API url: """

API_URL_PROMPT = PromptTemplate(
    input_variables=[
        "api_docs",
        "question",
    ],
    template=API_URL_PROMPT_TEMPLATE,
)

API_RESPONSE_PROMPT_TEMPLATE = (
    API_URL_PROMPT_TEMPLATE
    + """ {api_url}

#################################

Here is the response from the API:

{api_response}

#################################

Summarize this response to answer the original question.

Original question: {question}

Answer: """
)

API_RESPONSE_PROMPT = PromptTemplate(
    input_variables=["api_docs", "question", "api_url", "api_response"],
    template=API_RESPONSE_PROMPT_TEMPLATE,
)
