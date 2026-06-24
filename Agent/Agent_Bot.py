from typing import TypedDict,List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph ,START,END
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.7
)

class AgentState(TypedDict):
    message: List[HumanMessage]

def process(state:AgentState)->AgentState:
    reponse = llm.invoke(state['message'])
    print(f"\nAI: {reponse.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()




user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"message":[HumanMessage(content=user_input)]})

    user_input = input("Enter: ")