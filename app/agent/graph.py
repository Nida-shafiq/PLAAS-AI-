from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

from app.agent.state import AgentState
from app.agent.tools.faiss_tool import faiss_legal_search
from app.agent.tools.web_search_tool import web_legal_search
from app.agent.tools.disclaimer_tool import legal_disclaimer
from app.rag.config import GEMINI_API_KEY

SYSTEM_PROMPT = (
    "You are PLAAS AI, a legal research assistant specialized in Pakistani law "
    "(Constitution, Code of Criminal Procedure 1898, Police Order 2002). "
    "Always try faiss_legal_search first for Pakistani legal questions. "
    "Use web_legal_search only if the local database has no relevant result. "
    "Always call legal_disclaimer once at the end of a substantive legal answer. "
    "Cite sources (document, page, clause) whenever you use faiss_legal_search results."
)

tools = [faiss_legal_search, web_legal_search, legal_disclaimer]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)
llm_with_tools = llm.bind_tools(tools)


def call_model(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


agent_graph = build_graph()
