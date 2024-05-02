from dotenv import load_dotenv
load_dotenv(verbose=True)

import streamlit as st

import operator
from typing import TypedDict, Annotated, Sequence

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader

from langgraph.graph import StateGraph, END


# モデルのロード
model = ChatOpenAI(model_name='gpt-4-turbo')
# model = ChatOpenAI(model_name='gpt-3.5-turbo')

class AgentState(TypedDict):
    user_question : str #ユーザーの質問
    messages: Annotated[Sequence[BaseMessage], operator.add] #会話履歴
    knowledge_base : Annotated[str, operator.add] #知識ベース(検索した情報を追記していく)
    next_task_search_keyword : str #次に検索するキーワード
    next_task_search_content : str #次に検索して調べる内容
    answer_counter : Annotated[int, operator.add] #最終回答を試みた回数


def call_init_agentstate(state):
    state["user_question"] = state['messages'][-1].content
    state["messages"] = []
    state["knowledge_base"] = ""
    state["answer_counter"] = 0
    return state

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator

class SearchTask(BaseModel):
    search_reason: str = Field(description="検索する理由や目的。")
    search_keyword: str = Field(description="Wikipediaでの検索する単語。必ずひとつの単語である必要がある。")
    search_content : str = Field(description="検索したWikipediaのページで取得したい情報。")

gen_task_template ="""
以下の質問に回答するため、不足している情報をwikipediaから検索します。
検索する情報は最小の一つだけです。検索するための単語を記載してください。
検索する項目やその理由を記載してください。
検索する項目は一つだけです。
複数の質問がなされていても、一つの質問に対してのみ回答を生成してください。

{format_instructions}

-----------------
質問例:映画オッペンハイマーの監督が作成した映画の本数を調べてください。
search_reason="情報が不足しているため。とりあえずまず"映画オッペンハイマー"のwikipediaを検索して、監督の名前を調べる必要があります。"
search_keyword="オッペンハイマー"
search_content="映画オッペンハイマーの監督の名前"
-----------------
質問例:中性子と原子核が発見された年に起こった出来事を調べてください。
search_reason="情報が不足しているため、とりあえずまず"中性子"のwikipediaを検索して、とりあえず発見された年を調べる必要があります。"
search_keyword="中性子"
search_content="中性子の発見された年"
-----------------
質問例:中性子と原子核が発見された年に起こった出来事を調べてください。ただし、以下の情報はすでに検索済みです。中性子の発見された年は1932年です。
search_reason="原子核に関する情報が不足しているため、"原子核"のwikipediaを検索して、発見された年を調べる必要があります。"
search_keyword="原子核"
search_content="原子核の発見された年"
-----------------

質問:{user_question}。{knowledge_base}
"""
def call_generate_task(state):
    _parser = PydanticOutputParser(pydantic_object=SearchTask)
    _prompt = PromptTemplate(
        template=gen_task_template,
        input_variables=["user_question", "knowledge_base"],
        partial_variables={"format_instructions": _parser.get_format_instructions()},
    )
    chain = _prompt| model | _parser
    # chain = chain.with_retry(stop_after_attempt=5)
    res = chain.invoke({'user_question': state['user_question'], 'knowledge_base': state['knowledge_base']})

    return {
        "messages": [AIMessage(content=res.search_reason)],
        'next_task_search_keyword':res.search_keyword,
        'next_task_search_content':res.search_content
    }

def search_for_wikipedia(query: str) -> str:
    """
    Search for a wikipedia article and return the content of the first article found.
    """
    docs = WikipediaLoader(query=query, load_max_docs=5, lang='ja').load()
    if len(docs) > 0:
        return ''.join([d.page_content for d in docs])
    return ""


search_and_answer_template ="""
以下の参考文書を使用して、知りたい内容に回答してください。
知りたい内容には簡潔に数10文字のテキストで答えてください。
たとえば、「XXXはYYYです。」と短文で回答すること。

不要な情報は削除すること。

参考文書:{searched_content}

知りたい内容:{next_task_search_content}
"""

def call_search_and_answer(state):
    searched_content = search_for_wikipedia(state['next_task_search_keyword'])

    _parser = PydanticOutputParser(pydantic_object=SearchTask)
    _prompt = PromptTemplate(
        template=search_and_answer_template,
        input_variables=["next_task_search_content", "searched_content"],
    )
    chain = _prompt| model | StrOutputParser()

    res = chain.invoke({'next_task_search_content': state['next_task_search_content'], 'searched_content': searched_content})
    new_knowledge = f"検索した結果「{state['next_task_search_content']}」に関する情報は次の通り。\n{res}\n\n\n"
    return {
        "messages": [AIMessage(content=new_knowledge)],
        'knowledge_base':new_knowledge,
    }

final_answer ="""
以下の知識をもとに質問に回答してください。
もし回答に必要な情報が不足している場合は、回答できな理由を述べた後、「回答不可」と回答してください。


知識:{knowledge_base}

質問:{user_question}
"""
def call_final_answer(state):
    prompt = ChatPromptTemplate.from_template(template=final_answer)
    chain = prompt| model | StrOutputParser()
    res = chain.invoke({
        'user_question': state['user_question'],
        'knowledge_base': state['knowledge_base']
    })
    return {"messages": [AIMessage(content=res)], "answer_counter":1}

def router_fa(state):
    res = call_final_answer(state)
    if state['answer_counter'] >= 5:
        # print('回答制限')
        return 'end'
    if '回答不可' in res['messages'][-1].content:
        return 'continue'
    else:
        return 'end'
    
def get_runnable():
    workflow = StateGraph(AgentState)
    workflow.add_node("init_agent",call_init_agentstate)
    workflow.add_node("generate_task", call_generate_task)
    workflow.add_node("search_and_answer", call_search_and_answer)
    workflow.add_node("final_answer", call_final_answer)

    workflow.set_entry_point("init_agent")
    workflow.add_edge("init_agent", "generate_task")
    workflow.add_edge("generate_task", "search_and_answer")
    workflow.add_edge("search_and_answer", "final_answer")
    workflow.add_conditional_edges(
        "final_answer",
        router_fa,
        {
            "end": END,
            "continue": "generate_task"
        }
    )
    runnable = workflow.compile()
    return runnable

# ------------------------------------------- # -------------------------------------------
# ------------------------------------------- # -------------------------------------------
# ------------------------------------------- # -------------------------------------------
# ------------------------------------------- # -------------------------------------------
st.set_page_config(
    page_title="LangChain: Chat with search",
    page_icon="🦜",
    layout="wide", 
    initial_sidebar_state="auto"
)

st.title("🦜🕸️ LangGraph Agent: Chat with Wikipedia Search")



if "messages" not in st.session_state:
    st.session_state["messages"] = []

col1, col2 = st.columns(spec=[0.8,0.2])

with col1:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

with col2:
    st.markdown(f"### 知識ベース")


if prompt := st.chat_input():
    runnable = get_runnable()
    inputs = {"messages": [HumanMessage(content=prompt)]}
    with col1:
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    for output in runnable.stream(inputs):
        for key, value in output.items():
            if key == 'init_agent':
                response = f'{key}:エージェントの状態を初期化します。'
            else:
                response = f'{key}:' + value['messages'][-1].content
            avatar  = {
                'init_agent': '🦜',
                'generate_task': '📜',
                'search_and_answer': '🔍',
                'final_answer': '🤖',
            }[key]
            with col1:
                st.chat_message(name=key,avatar =avatar ).write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            with col2:
                new_knowledge_base = value.get('knowledge_base')
                if new_knowledge_base is not None:
                    st.markdown(f"* {new_knowledge_base}")