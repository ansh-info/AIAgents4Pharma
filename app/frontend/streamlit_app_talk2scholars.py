#!/usr/bin/env python3

"""
Talk2Scholars: A Streamlit app for the Talk2Scholars graph.
"""

import os
import sys
import random
import streamlit as st
import hydra
import pandas as pd
from streamlit_feedback import streamlit_feedback
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tracers.context import collect_runs
from langchain.callbacks.tracers import LangChainTracer
from langsmith import Client

sys.path.append("./")
from aiagents4pharma.talk2scholars.agents.main_agent2 import get_app

st.set_page_config(page_title="Talk2Scholars", page_icon="🤖", layout="wide")

# Initialize configuration
hydra.core.global_hydra.GlobalHydra.instance().clear()
if "config" not in st.session_state:
    # Load Hydra configuration
    with hydra.initialize(
        version_base=None,
        config_path="../../aiagents4pharma/talk2scholars/configs",
    ):
        cfg = hydra.compose(config_name="config", overrides=["app/frontend=default"])
        cfg = cfg.app.frontend
        st.session_state.config = cfg
else:
    cfg = st.session_state.config

# Check if env variable OPENAI_API_KEY exists
if "OPENAI_API_KEY" not in os.environ:
    st.error(
        "Please set the OPENAI_API_KEY environment \
        variable in the terminal where you run the app."
    )
    st.stop()

# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Welcome to Talk2Scholars!"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize project_name for Langsmith
if "project_name" not in st.session_state:
    # st.session_state.project_name = str(st.session_state.user_name) + '@' + str(uuid.uuid4())
    st.session_state.project_name = "Talk2Scholars-" + str(random.randint(1000, 9999))

# Initialize run_id for Langsmith
if "run_id" not in st.session_state:
    st.session_state.run_id = None

# Initialize graph
if "unique_id" not in st.session_state:
    st.session_state.unique_id = random.randint(1, 1000)
if "app" not in st.session_state:
    # st.session_state.app = get_app(st.session_state.unique_id)
    if "llm_model" not in st.session_state:
        st.session_state.app = get_app(st.session_state.unique_id)
    else:
        st.session_state.app = get_app(
            st.session_state.unique_id, llm_model=st.session_state.llm_model
        )

# Get the app
app = st.session_state.app


def _submit_feedback(user_response):
    """
    Function to submit feedback to the developers.
    """
    client = Client()
    client.create_feedback(
        st.session_state.run_id,
        key="feedback",
        score=1 if user_response["score"] == "👍" else 0,
        comment=user_response["text"],
    )
    st.info("Your feedback is on its way to the developers. Thank you!", icon="🚀")


@st.dialog("Warning ⚠️")
def update_llm_model():
    """
    Function to update the LLM model.
    """
    llm_model = st.session_state.llm_model
    st.warning(
        f"Clicking 'Continue' will reset all agents, \
            set the selected LLM to {llm_model}. \
            This action will reset the entire app, \
            and agents will lose access to the \
            conversation history. Are you sure \
            you want to proceed?"
    )
    if st.button("Continue"):
        # Delete all the messages and the app key
        for key in st.session_state.keys():
            if key in ["messages", "app"]:
                del st.session_state[key]


# Main layout of the app split into two columns
main_col1, main_col2 = st.columns([3, 7])
# First column
with main_col1:
    with st.container(border=True):
        # Title
        st.write(
            """
            <h3 style='margin: 0px; padding-bottom: 10px; font-weight: bold;'>
            🤖 Talk2Scholars
            </h3>
            """,
            unsafe_allow_html=True,
        )

        # LLM panel (Only at the front-end for now)
        llms = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        # llm_option = st.selectbox(
        #     "Pick an LLM to power the agent",
        #     llms,
        #     index=0,
        #     key="st_selectbox_llm"
        # )
        st.selectbox(
            "Pick an LLM to power the agent",
            llms,
            index=0,
            key="llm_model",
            on_change=update_llm_model,
        )

        # Upload files (placeholder)
        # uploaded_file = st.file_uploader(
        #     "Upload sequencing data",
        #     accept_multiple_files=False,
        #     type=["h5ad"],
        #     help='''Upload a single h5ad file containing the sequencing data.
        #     The file should be in the AnnData format.'''
        #     )

    with st.container(border=False, height=500):
        prompt = st.chat_input("Say something ...", key="st_chat_input")

# Second column
with main_col2:
    # Chat history panel
    with st.container(border=True, height=575):
        st.write("#### 💬 Chat History")

        # Display chat messages
        for count, message in enumerate(st.session_state.messages):
            if message["type"] == "message":
                with st.chat_message(
                    message["content"].role,
                    avatar="🤖" if message["content"].role != "user" else "👩🏻‍💻",
                ):
                    st.markdown(message["content"].content)
                    st.empty()
            elif message["type"] == "dataframe":
                if 'tool_name' in message:
                    if message['tool_name'] == 'display_results':
                        df_papers = message["content"]
                        st.dataframe(df_papers,
                                    use_container_width=True,
                                    key=message["key"],
                                    hide_index=True,
                                    column_config={
                                        "URL": st.column_config.LinkColumn(
                                            display_text="Open URL",
                                        ),
                                    }
                                )
                # else:
                #     streamlit_utils.render_table(message["content"],
                #                     key=message["key"],
                #                     # tool_name=message["tool_name"],
                #                     save_table=False)
                st.empty()

        # When the user asks a question
        if prompt:
            # Create a key 'uploaded_file' to read the uploaded file
            # if uploaded_file:
            #     st.session_state.article_pdf = uploaded_file.read().decode("utf-8")

            # Display user prompt
            prompt_msg = ChatMessage(prompt, role="user")
            st.session_state.messages.append({"type": "message", "content": prompt_msg})
            with st.chat_message("user", avatar="👩🏻‍💻"):
                st.markdown(prompt)
                st.empty()

            with st.chat_message("assistant", avatar="🤖"):
                # with st.spinner("Fetching response ..."):
                with st.spinner():
                    # Get chat history
                    history = [
                        (m["content"].role, m["content"].content)
                        for m in st.session_state.messages
                        if m["type"] == "message"
                    ]
                    # Convert chat history to ChatMessage objects
                    chat_history = [
                        (
                            SystemMessage(content=m[1])
                            if m[0] == "system"
                            else (
                                HumanMessage(content=m[1])
                                if m[0] == "human"
                                else AIMessage(content=m[1])
                            )
                        )
                        for m in history
                    ]

                    # Create config for the agent
                    config = {"configurable": {"thread_id": st.session_state.unique_id}}

                    # Update the agent state with the selected LLM model
                    current_state = app.get_state(config)
                    app.update_state(config, {"llm_model": st.session_state.llm_model})

                    # with collect_runs() as cb:
                    #     # Add Langsmith tracer
                    #     tracer = LangChainTracer(
                    #         project_name=st.session_state.project_name
                    #     )
                    ######################################################
                    # Get response from the agent
                    # response = app.invoke(
                    #     {"messages": [HumanMessage(content=prompt)]},
                    #     config=config | {"callbacks": [tracer]},
                    # )
                    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config,
                    )
                    #######################################################
                        # st.session_state.run_id = cb.traced_runs[-1].id
                    # Print the response
                    # print (response["messages"][-1])
                    current_state = app.get_state(config)
                    # print (current_state.values["messages"])

                    # Add assistant response to chat history
                    assistant_msg = ChatMessage(
                        response["messages"][-1].content, role="assistant"
                    )
                    st.session_state.messages.append(
                        {"type": "message", "content": assistant_msg}
                    )
                    # Display the response in the chat
                    st.markdown(response["messages"][-1].content)
                    st.empty()
                    reversed_messages = current_state.values["messages"][::-1]
                    # Loop through the reversed messages until a 
                    # HumanMessage is found i.e. the last message 
                    # from the user. This is to display the results
                    # of the tool calls made by the agent since the
                    # last message from the user.
                    for msg in reversed_messages:
                        # print (msg)
                        # Break the loop if the message is a HumanMessage
                        # i.e. the last message from the user
                        if isinstance(msg, HumanMessage):
                            break
                        # Skip the message if it is an AIMessage
                        # i.e. a message from the agent. An agent
                        # may make multiple tool calls before the
                        # final response to the user.
                        if isinstance(msg, AIMessage):
                            # print ('AIMessage', msg)
                            continue
                        # Work on the message if it is a ToolMessage
                        # These may contain additional visuals that
                        # need to be displayed to the user.
                        # print("ToolMessage", msg)
                        # Skip the Tool message if it is an error message
                        if msg.status == "error":
                            continue
                        # print("ToolMessage", msg)
                        uniq_msg_id = '_'.join([msg.name, msg.tool_call_id, str(st.session_state.run_id)])
                        # if msg.name in ['search_tool',
                        #                 'get_single_paper_recommendations',
                        #                 'get_multi_paper_recommendations']:
                        if msg.name in ['display_results']:
                            # Display the results of the tool call
                            # for msg_artifact in msg.artifact:
                            # dic_papers = msg.artifact[msg_artifact]
                            dic_papers = msg.artifact
                            if not dic_papers:
                                continue
                            df_papers = pd.DataFrame.from_dict(dic_papers, orient='index')
                            # Drop colum abstract
                            df_papers.drop(columns=['Abstract'], inplace=True)
                            st.dataframe(df_papers,
                                hide_index=True,
                                column_config={
                                    "URL": st.column_config.LinkColumn(
                                        display_text="Open URL",
                                    ),
                                }
                            )
                            # Add data to the chat history
                            st.session_state.messages.append({
                                    "type": "dataframe",
                                    "content": df_papers,
                                    "key": "dataframe_"+uniq_msg_id,
                                    "tool_name": msg.name
                                })
                            st.empty()
        # Collect feedback and display the thumbs feedback
        if st.session_state.get("run_id"):
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                on_submit=_submit_feedback,
                key=f"feedback_{st.session_state.run_id}",
            )
