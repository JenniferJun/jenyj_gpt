import json
import streamlit as st
from typing_extensions import override
from openai import AssistantEventHandler
import openai as client
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader

ASSISTANT_NAME = "Research Assistant"

class EventHandler(AssistantEventHandler):
    message = ""

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    def on_text_delta(self, delta, snapshot):
        self.message += delta.value
        self.message_box.markdown(self.message.replace("$", "\$"))

    def on_event(self, event):
        print(event.event)
        if event.event == "thread.run.requires_action":
            submit_tool_outputs(event.data.id, event.data.thread_id)

st.set_page_config(
    page_title="Research Assistant GPT",
    page_icon="🧰",
)

st.title("Research Assistant GPT")

st.markdown(
    """
    Welcome to Research Assistant GPT.
            
     Ask a question for research and our Assistant will support reference URLs and summarise those pages.
"""
)


# Tools
def get_term(inputs):
    wkp = WikipediaAPIWrapper()
    query = inputs["query"]
    return wkp.run(f"Term name of {query}")


def get_urls(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    term_name = inputs["term_name"]
    return ddg.run(f"Referrence URLs of {term_name}")

def extract_urls(urls):
    web_loader = WebBaseLoader(urls) 
    loaded_contents = web_loader.load()   
    return loaded_contents
 


functions_map = {
    "get_term": get_term,
    "get_urls": get_urls,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_term",
            "description": "Given query returns its term user want to know",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to research. Example: Research about Donald Trump",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_urls",
            "description": "Given the term returns only URLs of those research",
            "parameters": {
                "type": "object",
                "properties": {
                    "term_name": {
                        "type": "string",
                        "description": "The term that you want to know",
                    }
                },
                "required": ["term_name"],
            },
        },
    }, 
        {
        "type": "function",
        "function": {
            "name": "extract_urls",
            "description": "Extracts web content from URLs",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "string",
                        "description": "URLs from get_urls",
                    }
                },
                "required": ["urls"],
            },
        },
    }, 
]

#### Utilities
def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    with client.beta.threads.runs.submit_tool_outputs_stream(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


def insert_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def paint_history(thread_id):
    messages = get_messages(thread_id)
    for message in messages:
        insert_message(
            message.content[0].text.value,
            message.role,
        )


openapi_key = st.sidebar.text_input("OpenAI API KEY : ")
if not openapi_key:
    st.error("Please enter your OpenAI API key to proceed.")
else:
    if "assistant" not in st.session_state:
        client.api_key = openapi_key
        assistants = client.beta.assistants.list(limit=10)
        for a in assistants:
            if a.name == ASSISTANT_NAME:
                assistant = client.beta.assistants.retrieve(a.id)
                break
        else:
            assistant = client.beta.assistants.create(
                name=ASSISTANT_NAME,
                instructions="You help users do research on the given query using search engines. You give users the summarization of the information you got.",
                model="gpt-4o-mini",
                tools=functions,
            )
        thread = client.beta.threads.create()
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread
    else:
        assistant = st.session_state["assistant"]
        thread = st.session_state["thread"]

    paint_history(thread.id)
    content = st.chat_input("What do you want to know?")
    if content:
        with st.spinner('Researching...'):
            send_message(thread.id, content)
            insert_message(content, "user")

            with st.chat_message("assistant"):
                with client.beta.threads.runs.stream(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                    event_handler=EventHandler(),
                ) as stream:
                    stream.until_done()

