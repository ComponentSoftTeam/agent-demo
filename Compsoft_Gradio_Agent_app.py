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

# +
from langsmith import Client

import os

client = Client()
# -

News_api_key = os.environ["NEWS_API_KEY"]

if os.environ["GRADIO_USER"]:
    Gradio_user = os.environ["GRADIO_USER"]
    Gradio_password = os.environ["GRADIO_PASSWORD"]

# + active=""
# if os.environ["GCP"]:
#     !gcloud auth application-default login
# -

from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from local_google_genai_chat_models import ChatGoogleGenerativeAI
#from langchain_google_vertexai import ChatVertexAI
from local_google_vertexai_chat_models import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_fireworks.chat_models import ChatFireworks
from langchain_core.output_parsers.string import StrOutputParser

# +
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

#from langchain.agents import create_tool_calling_agent, create_react_agent, AgentExecutor
from langchain.agents import create_tool_calling_agent, create_react_agent
from local_agent import AgentExecutor
#from langchain_community.agent_toolkits.load_tools import load_tools
from local_load_tools import load_tools
from langchain_core.prompts import ChatPromptTemplate
from langchain.globals import set_debug, set_verbose
from langchain_core.output_parsers.string import StrOutputParser
from langchain import hub

set_debug(False)
set_verbose(False)

# +
from typing import TYPE_CHECKING, Any, Dict, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish

"""Global list variable which will be displayed in the Agent's thoughts Gradio frame."""
trace_list: dict[str, list[str]] = {}

"""Function used by VariableCallbackHandler that appends new callback outputs to the trace_list global list variable which will be displayed in the Agent's thoughts Gradio frame."""
def trace_list_append(session_id: str, text: str):
    if session_id not in trace_list:
        trace_list[session_id] = []     
    trace_list[session_id].append(text)

"""Callback Handler that writes to a variable."""
class VariableCallbackHandler(BaseCallbackHandler):
    """Callback Handler that prints to a variable."""

    def __init__(self,  session_id:str, color: Optional[str] = None) -> None:
        super().__init__()
        self._session_id = session_id

    """def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        trace_list_append(self._session_id,  f">> ENTERING new {class_name} CHAIN\n")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        outputs = str(outputs)
        kwargs = str(kwargs)
        trace_list_append(self._session_id,  f">> FINISHED ??? CHAIN.\nOUTPUTS = {outputs}\nKWARGS = {kwargs}")"""

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        if color == "blue":
            new_text = f"\nREASONING: Tool to use: {action.tool}, Tool input: {action.tool_input}"
        else:
            action_log = action.log.strip("\n ")
            new_text = f"\nACTION: {action_log}"
        trace_list_append(self._session_id,  f"{new_text}")

    def on_tool_end(
        self,
        output: Any,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        output = output.strip("\n ")
        #if observation_prefix is not None:
        #    trace_list_append(self._session_id,  f"\nobservation_prefix = {observation_prefix}\n")        
        trace_list_append(self._session_id, f"\nOBSERVATION:  {output}")
        #if llm_prefix:
        #    trace_list_append(self._session_id,  f"\nllm_prefix = {llm_prefix}\n")

    """def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        trace_list_append(self._session_id,  f"{text}\n")"""

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        new_text = f"\nFINAL ANSWER: {finish.log}"
        trace_list_append(self._session_id,  new_text)



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
        #"gemini-1.5-flash": ChatVertexAI(            
            model="gemini-1.5-flash",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        ),
        "gemini-1.5-pro": ChatGoogleGenerativeAI(
        #"gemini-1.5-pro": ChatVertexAI(
            model="gemini-1.5-pro",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        ),
        "claude-3-haiku": ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        ),    
        "claude-3-sonnet": ChatAnthropic(
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
    llm = LLM_MODELS[model_type]
      
    # Model output -> string
    parser = StrOutputParser()
    #varcallhandler = VariableCallbackHandler()
    #stdoutcallhandler = StdOutCallbackHandler()    
    
    llm_math_tool, weather_tool, news_tool, wikipedia_tool, arxiv_tool, websearch_tool = load_tools(["llm-math", "open-meteo-api", "news-api", "wikipedia", "arxiv", "ddg-search"], news_api_key=News_api_key, llm=llm)
    #llm_math_tool.invoke("What is the square root of 4?", verbose=True)
    #wikipedia_tool.invoke("How many people live in Prague?", verbose=True)
    #print(weather_tool.invoke(tool_prompt, verbose=True))
    #print(news_tool.invoke("What are the 10 most important news in Prague? Answer in 10 bullet points"))
    #print(arxiv_tool.invoke("List the title of 10 scientific papers about LLM agents published in this year.", verbose=True))
    #print(websearch_tool.invoke("Who won the most Oscar in this year?"))
    
    tools=[llm_math_tool, news_tool, weather_tool, arxiv_tool, wikipedia_tool, websearch_tool]
    
    if model_type.endswith("prompting"):
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

    
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose = False, stream_runnable=False)
    agent_chain = agent_executor | itemgetter("output")
    
    return agent_chain

# +
modelfamilies_model_dict = {
    "OpenAI GPT": ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
    "Google Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],  
    "Anthropic Claude": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
    "MistralAI Mistral": ["mistral-large", "open-mixtral-8x22b", "mistral-small"],
    "Meta Llama" : ["Llama-3-70b-firefunction-v2", "llama-3-70b-prompting"],
}

def generate_system_prompt():
    #return f"You're a helpful assistant. Always use tools to answer questions. Always use the Calculator for calculations, even when adding 2 numbers or calculating averages. The current date is {datetime.today().strftime('%Y-%m-%d')}."
    return f"You're a helpful assistant. Always use tools to answer questions. Always use the Calculator for calculations, even when adding 2 numbers or calculating averages. The current date is {datetime.today().strftime('%Y-%m-%d')}"

prompt_text = "What is the square root of 4?"
#prompt = f"Create a table that contains the date as well as the maximum, minimum and average temperature values as well as the sum of precipitations in the coming 7 days in Budapest"
#prompt = f"""Create a table that contains the date as well as the maximum, minimum and average temperature values as well as the sum of precipitations separately for each of the coming 7 days in Budapest.
#Then include a line with each of the average of the maximum, minimum and avarage temperature values for these 7 days. And after all that also write a list with 10 current news in Budapest Hungary."""
trace = ""




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
    text = f"PROMPT:  {prompt}"
    trace_list_append(session_id, text)
    chat = chatbot or []
    chat.append([prompt, ""])
    trace = ""    
    
    agent_chain = get_chain(session_id, model_type=model_type)
    response = agent_chain.invoke({"system_prompt": system_prompt, "input": prompt}, {"callbacks": [VariableCallbackHandler(session_id)]})
    chat[-1][1] = response
  
    return chat, ""

def exec_agent_streaming(chatbot, system_prompt ="", prompt="I have no request", model_type="mistral-large-latest"):
    raise NotImplementedError()
    # global trace_list
    # trace_list.clear()
    # text = f"PROMPT:  {prompt}"
    # trace_list_append(session_id, text)
    # chat = chatbot or []
    # chat.append([prompt, ""])
    # trace = ""
    
    # agent_chain = get_chain(model_type=model_type)
    # response = agent_chain.stream({"input": prompt}, {"callbacks": [VariableCallbackHandler(session_id)]})

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

with gr.Blocks(title="CompSoft") as demo:
    session_id = gr.State(value=generate_session_id)
    gr.Markdown("# Component Soft Agent Demo (Calculator, Websearch, Wikipedia, Arxiv, Weather, News)")
    system_prompt = gr.Textbox(label="System prompt", value=generate_system_prompt)
    with gr.Row():
        modelfamily = gr.Dropdown(list(modelfamilies_model_dict.keys()), label="Model family", value="OpenAI GPT")
        model_type = gr.Dropdown(list(modelfamilies_model_dict["OpenAI GPT"]), label="Model", value="gpt-4o")       
    with gr.Row():
        thoughts=gr.Textbox(label="Agent's thoughts", value="", interactive=False, lines=13, max_lines=13)
        chatbot=gr.Chatbot(label="Agent's answer", height=325, show_copy_button=True, scale = 2)
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value=prompt_text)
    with gr.Row():
        submit_btn_nostreaming = gr.Button("Answer")
        clear_btn = gr.ClearButton([prompt, chatbot, thoughts])

    rep = demo.load(thoughts_func, inputs=[session_id], outputs=thoughts, every=1)


    @modelfamily.change(inputs=modelfamily, outputs=[model_type])
    def update_modelfamily(modelfamily):
        model_type = list(modelfamilies_model_dict[modelfamily])
        return gr.Dropdown(choices=model_type, value=model_type[0], interactive=True)

    submit_btn_nostreaming.click(exec_agent, inputs=[chatbot,session_id, system_prompt, prompt, model_type], outputs=[chatbot, prompt])
    clear_btn.click(clear_texts, inputs=[session_id], outputs=[prompt, chatbot, thoughts], )
    
    gr.Examples(
        ["What is the square root of 4?", "How many people live in Budapest?", "What will be the weather like tomorrow in Budapest?", "What are the 10 most important news in Budapest? Answer in 10 bullet points.",  
         "Who won the most Oscars in this year?", "List the title of 10 scientific papers about LLM agents published in this year.", 
         "Create a table that contains the date as well as the maximum, minimum and average temperature values as well as the sum of precipitations separately for each the coming 3 days in Budapest?", 
         "Create a table that contains the date as well as the maximum, minimum and average temperature values as well as the sum of precipitations separately for each of the coming 7 days in Budapest. Then include a line with each of the average of the maximum, minimum and avarage temperature values for these 7 days. And after all that also write a list with 10 current news in Budapest Hungary.",
        ],
        prompt
    )

#demo.launch()
demo.launch(share=True)
#demo.launch(share=True, share_server_address="gradio.componentsoft.ai:7000", share_server_protocol="https", auth=(Gradio_user, Gradio_password), max_threads=20, show_error=True, favicon_path="data/favicon.ico", state_session_capacity=20)
# -












