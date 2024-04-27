from dotenv import load_dotenv
load_dotenv(verbose=True)
# import os
# os.environ['ANTHROPIC_API_KEY'] = 'your-api-key'
# os.environ['TAVILY_API_KEY] = 'hogehoge'

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langchain.memory import ConversationBufferMemory
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

st.set_page_config(page_title="LangChain: Chat with search", page_icon="🦜")
st.title("🦜 LangChain: Chat with search")

with st.sidebar:
    st.markdown("### Chat history")
    st.write("The chat history will be reset if there are no messages or if the button is pressed.")

# msgs :userとaiの会話履歴(中間ステップは含まない)
# memory :よくわからん。とりあえずmsgをmemoryにいれ、memoryをagentexecuterに渡しておくと、agente実行時にmsgの内容が更新されるっぽい。
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    # どういうわけか、最初をaiメッセージにするとエラーになる。なので、userメッセージを最初にする。
    msgs.add_user_message("こんにちは。これから質問しますので日本語で答えてください。")
    msgs.add_ai_message("承知しました。何を調べますか？")
    # 中間ステップを保存するための辞書を定義(key:付属のメッセージのインデックス, value:中間ステップリスト)
    st.session_state.intermediate_steps = {}


for idx, msg in enumerate(msgs.messages):
    # st.chat_messageでユーザー/AIの回答を表示する
    with st.chat_message(msg.type): # user or assistant, https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
        for intermediate_step in st.session_state.intermediate_steps.get(str(idx), []):
            agent_action, agent_output = intermediate_step
            # 01.中間ステップの内容をステータスコンテナに記載して、、、
            with st.status(f"**{agent_action.tool}**: {agent_action.tool_input}", state="complete"): # https://docs.streamlit.io/develop/api-reference/status/st.status
                st.write(agent_action.log)
                st.write(agent_output)
        # 02.その下にメッセージを記載する。
        st.write(msg.content)


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    model = ChatAnthropic(streaming=True,model='claude-3-haiku-20240307')
    tools = [TavilySearchResults(max_results=1)]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=model, tools=tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant"):
        # この辺もよくわからん。
        # 最終回答の直前に「Complete!」と記載されたステータスコンテナが表示されるのも、ここの処理のためだと思うが、よくわからん。
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        response = executor.invoke(prompt, cfg)

        # 最終回答を表示
        st.write(response["output"])

        # 中間ステップを保存
        st.session_state.intermediate_steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]