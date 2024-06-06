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
#import pandas as pd
#from pprint import pprint
from operator import itemgetter
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# +
from langsmith import Client

import os
os.environ["LANGCHAIN_PROJECT"] = "Agents with Langchain"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_TRACING_V2"] = "false"

client = Client()
# -

News_api_key = os.environ["NEWS_API_KEY"]
# Financial_news_api_key=os.environ["ALPHAVANTAGE_API_KEY"]

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_fireworks.chat_models import ChatFireworks

# +
#from langchain.tools import WikipediaQueryRun

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

from langchain.agents import create_tool_calling_agent, create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.prompts import ChatPromptTemplate
from langchain.globals import set_debug, set_verbose
from langchain_core.output_parsers.string import StrOutputParser

from langchain import hub

set_debug(False)
set_verbose(False)


# +
"""Callback Handler that writes to a variable."""



trace_list: dict[str, list[str]] = {}

def trace_list_append(session_id: str, text: str):
    if session_id not in trace_list:
        trace_list[session_id] = []
    trace_list[session_id].append(text)

#from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from langchain_core.callbacks.base import BaseCallbackHandler 
from langchain_core.callbacks.stdout import StdOutCallbackHandler

#if TYPE_CHECKING:
from langchain_core.agents import AgentAction, AgentFinish

class VariableCallbackHandler(BaseCallbackHandler):
    """Callback Handler that prints to a variable."""

    def __init__(self,  session_id:str, color: Optional[str] = None) -> None:
        super().__init__()
        self._session_id = session_id
    

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        trace_list_append(self._session_id,  f">> ENTERING AgentExecutor CHAIN\n")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        trace_list_append(self._session_id,  f">> FINISHED AgentExecutor CHAIN.")

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        action_log = action.log.strip("\n ")
        new_text = f"REASONING: {action_log}\n"
        trace_list_append(self._session_id,  f"\n{new_text}")
        #kwarg = str(kwargs)
        #trace_list_append(self._session_id,  kwarg)

    def on_tool_start(
        self,
        **kwargs: Any,
    ) -> None:
        kwarg = str(kwargs)
        trace_list_append(self._session_id,  kwarg)

    def on_tool_end(
        self,
        outputs: Any,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            trace_list_append(self._session_id,  f"\nobservation_prefix = {observation_prefix}\n")        
        if outputs:
            trace_list_append(self._session_id, f"\noutput = {str(outputs)}\n")
        if llm_prefix:
            trace_list_append(self._session_id,  f"\nllm_prefix = {llm_prefix}\n")

    """def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        token1 = str(token)
        trace_list_append(self._session_id,  token1)

    def _on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        response1 = str(response)
        trace_list_append(self._session_id,  response1)"""

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        newtext = f"{text}\n"
        trace_list_append(self._session_id,  newtext)

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        #print_text(finish.log, color=color or self.color, end="\n")
        #new_text = print_text(finish.log, color=color or self.color, end="\n")
        new_text = f"\nFINAL ANSWER: {finish.log}\n"
        trace_list_append(self._session_id,  new_text)
        #kwarg = str(kwargs)
        #trace_list_append(self._session_id,  kwarg)

# -

def get_chain(session_id: str, model_type="mistral-large-latest"):
    
    TEMPERATURE = 0.0
    MAX_NEW_TOKENS = 4000

    LLM_MODELS = {
        "llama-3-70b-prompting": ChatFireworks(
            model_name="accounts/fireworks/models/llama-v3-70b-instruct",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        ),
        "Llama-2-70b-firefunction-v1": ChatFireworks(
            model_name="accounts/fireworks/models/firefunction-v1",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,            
        ),
        "gpt-3.5-turbo": ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        ), 
        "gpt-4-turbo": ChatOpenAI(
            model="gpt-4-turbo",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        ), 
        "gpt-4o": ChatOpenAI(
            model="gpt-4o",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        ),
        "mistral-large-latest": ChatMistralAI(
            model="mistral-large-latest",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        ),
        "mistral-small-latest": ChatMistralAI(
            model="mistral-small-latest",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        ),        
        "open-mixtral-8x22b": ChatMistralAI(
            model="open-mixtral-8x22b",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        ),
        "gemini-1.5-flash-latest": ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
            #convert_system_message_to_human=True
        ),        
        "gemini-1.5-pro-latest": ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
            #convert_system_message_to_human=True
        )
    }
    llm = LLM_MODELS[model_type]
      
    # Model output -> string
    parser = StrOutputParser()
    
    #tool_prompt = f"Create a table that contains the date as well as the maximum, minimum and avarage temperature values separately for each of 2024-05-18, 2024-05-19 and 2024-05-20 in Budapest."
    tool_prompt = f"Create a table that contains the date as well as the maximum, minimum and avarage temperature values as well as the sum of precipitations separately for each of the coming 7 days in Budapest"
    prompt_text = "What are the maximum, minimum and average temperature values as well as the sum of precipitations tomorrow in Budapest?"

    
    #pubmed_tool = PubmedQueryRun()
    #llm_math_tool, wikipedia_tool, weather_tool, news_tool = load_tools(["llm-math","wikipedia", "open-meteo-api", "news-api"], news_api_key=News_api_key, llm=llm)
    llm_math_tool, websearch_tool, wikipedia_tool, arxiv_tool, weather_tool, news_tool = load_tools(["llm-math","ddg-search", "wikipedia", "arxiv", "open-meteo-api", "news-api"], news_api_key=News_api_key, llm=llm)
    #llm_math_tool.invoke("What is the square root of 4?", verbose=True)
    #wikipedia_tool.invoke("How many people live in Prague?", verbose=True)
    #print(weather_tool.invoke(tool_prompt, verbose=True))
    #print(news_tool.invoke("What are the 10 most important news in Prague? Answer in 10 bullet points"))
    #print(arxiv_tool.invoke("List the title of 10 scientific papers about LLM agents published in this year.", verbose=True))
    #print(websearch_tool.invoke("Who won the most Oscar in this year?"))
    #print(pubmed_tool.invoke("List papers about Vortioxetin."))
    
    tools=[llm_math_tool, news_tool, weather_tool, arxiv_tool, wikipedia_tool, websearch_tool] # More tools could be added
    # Be careful with older tools, they might break with newer models
    
    print(model_type)
    
    if model_type == "llama-3-70b-prompting":
        prompt_react = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt_react)
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])        
        agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, callbacks=[VariableCallbackHandler(session_id)], verbose = False, stream_runnable=False)
    #agent_executor = AgentExecutor(agent=agent, tools=tools, verbose = True)
    # , enable_automatic_function_calling=True
    agent_chain = agent_executor | itemgetter("output")
    
    return agent_chain

# +
modelfamilies_model_dict = {
    "OpenAI GPT": ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
    "Google Gemini": ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"],    
    "MistralAI Mistral": ["mistral-large-latest", "open-mixtral-8x22b", "mistral-small-latest"],
    "Meta Llama" : ["llama-3-70b-prompting", "Llama-2-70b-firefunction-v1"],
}

def generate_system_prompt():
    #return f"You're a helpful assistant. Always use tools to answer questions. Always use the Calculator for calculations, even when adding 2 numbers or calculating averages. The current date is {datetime.today().strftime('%Y-%m-%d')}."
    return f"You're a helpful assistant. Always use tools to answer questions. Always use the Calculator for calculations, even when adding 2 numbers or calculating averages. The current date is {datetime.today().strftime('%Y-%m-%d')}"

prompt_text = "What is the square root of 4?"
#prompt = f"Create a table that contains the date as well as the maximum, minimum and average temperature values as well as the sum of precipitations in the coming 7 days in Budapest"
#prompt = f"""Create a table that contains the date as well as the maximum, minimum and average temperature values as well as the sum of precipitations separately for each of the coming 7 days in Budapest.
#Then include a line with each of the average of the maximum, minimum and avarage temperature values for these 7 days. And after all that also write a list with 10 current news in Budapest Hungary."""
trace = ""
#thoughts_string = "You will see here the agent's thoughts to answer the request."
#stdoutcallhandler = StdOutCallbackHandler()




# +
def clear_texts(session_id):
    global trace_list
    
    if session_id in trace_list:
        del trace_list[session_id]

    return "", [], ""

def thoughts_func(session_id) -> str | None:
    # print()
    global trace_list
    if session_id not in trace_list:
        return ""
    
    #print(f"The session id is: {session_id}")
    trace = ""
    for trace_item in trace_list[session_id]:
        #print(f"{i}, {trace_item}")
        #trace = trace + convert_to_markdown(trace_item)
        trace = trace + trace_item

    return trace

def exec_agent(chatbot, session_id: str, system_prompt ="", prompt="I have no request", model_type="mistral-large-latest"):
    clear_texts(session_id)
    #trace_list_append(self._session_id,  "**Agent's thoughts:**")
    chat = chatbot or []
    chat.append([prompt, ""])
    trace = ""    
    
    agent_chain = get_chain(session_id, model_type=model_type)
    response = agent_chain.invoke({"input": prompt, "system_prompt": system_prompt})

    #print([response])

    # trace = ""
    # for i, trace_item in enumerate(trace_list):
    #     #print(f"{i}, {trace_item}")
    #     trace = trace + trace_item

    chat[-1][1] = response
  
    return chat, ""

def exec_agent_streaming(chatbot, system_prompt ="", prompt="I have no request", model_type="mistral-large-latest"):
    raise NotImplementedError()
    # global trace_list
    # trace_list.clear()
    # #trace_list_append(self._session_id,  f"\n> *Finished chain with output: {outputs}.*")
    # chat = chatbot or []
    # chat.append([prompt, ""])
    # trace = ""
    
    # agent_chain = get_chain(model_type=model_type)
    # response = agent_chain.stream({"input": prompt, "system_prompt": system_prompt})

    # for res in response:
    #     if res is not None:
    #         chat[-1][1] += res
    
    # for i, trace_item in enumerate(trace_list):
    #     #print(f"{i}, {trace_item}")
    #     # trace = trace + trace_item
        
    #     yield chat, ""



# +
import random
from time import sleep
import uuid
import gradio as gr
trace = ""

gr.close_all()

def generate_session_id():
    return str(uuid.uuid4())

#callback = gr.CSVLogger()

with gr.Blocks(title="CompSoft") as demo:
    session_id = gr.State(value=generate_session_id)
    gr.Markdown("# Component Soft Agent Demo (Calculator, Websearch, Wikipedia, Arxiv, Weather, News)")
    system_prompt = gr.Textbox(label="System prompt", value=generate_system_prompt)
    with gr.Row():
        modelfamily = gr.Dropdown(list(modelfamilies_model_dict.keys()), label="Model family", value="OpenAI GPT")
        model_type = gr.Dropdown(list(modelfamilies_model_dict["OpenAI GPT"]), label="Model", value="gpt-4o")       
    with gr.Row():
        thoughts=gr.Textbox(label="Agent's thoughts", value="", interactive=False, lines=13, max_lines=13)
        #thoughts=gr.Markdown(label="Agent's thoughts", value=thoughts_func(), header_links=False)
        #chatbot=gr.Chatbot(label="Agent's answer", height=325, show_copy_button=True, placeholder = "You'll see the agents' answer here", scale = 2)
        chatbot=gr.Chatbot(label="Agent's answer", height=325, show_copy_button=True, scale = 2)
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value=prompt_text)
    with gr.Row():
        submit_btn_nostreaming = gr.Button("Answer")
        #submit_btn_streaming = gr.Button("Answer with streaming")
        clear_btn = gr.ClearButton([prompt, chatbot, thoughts])
        #flag_btn = gr.Button("Flag")

    rep = demo.load(thoughts_func, inputs=[session_id], outputs=thoughts, every=1)


    @modelfamily.change(inputs=modelfamily, outputs=[model_type])
    def update_modelfamily(modelfamily):
        model_type = list(modelfamilies_model_dict[modelfamily])
        return gr.Dropdown(choices=model_type, value=model_type[0], interactive=True)

    #submit_btn_streaming.click(exec_agent_streaming, inputs=[chatbot, system_prompt, prompt, model_type], outputs=[chatbot, prompt])
    submit_btn_nostreaming.click(exec_agent, inputs=[chatbot,session_id, system_prompt, prompt, model_type], outputs=[chatbot, prompt])
    clear_btn.click(clear_texts, inputs=[session_id], outputs=[prompt, chatbot, thoughts], )

    #callback.setup([modelfamily, model_type, chatbot], "flagged_data_points")
    #flag_btn.click(lambda *args: callback.flag(args), [modelfamily, model_type, chatbot], None, preprocess=False)
    
    gr.Examples(
        ["What is the square root of 4?", "How many people live in Budapest?", "What will be the weather like tomorrow in Budapest?", "What are the 10 most important news in Budapest? Answer in 10 bullet points.",  
         "Who won the most Oscars in this year?", "List the title of 10 scientific papers about LLM agents published in this year.", 
         "Create a table that contains the date as well as the maximum, minimum and average temperature values as well as the sum of precipitations separately for each the coming 3 days in Budapest?", 
         "Create a table that contains the date as well as the maximum, minimum and average temperature values as well as the sum of precipitations separately for each of the coming 7 days in Budapest. Then include a line with each of the average of the maximum, minimum and avarage temperature values for these 7 days. And after all that also write a list with 10 current news in Budapest Hungary.",
        ],
        prompt
    )

#demo.launch()
demo.launch(share=True, share_server_address="gradio.componentsoft.ai:7000", share_server_protocol="https", auth=("CompSoft", "Bikszadi16"), max_threads=20, show_error=True, favicon_path="data/favicon.ico", state_session_capacity=20)
# -










