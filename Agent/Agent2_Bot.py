from typing import TypedDict,List, Union
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph ,START,END
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.7
)

class AgentState(TypedDict):
    message: List[Union[HumanMessage,AIMessage]]

def process(state:AgentState)->AgentState:
    response = llm.invoke(state['message'])
    state['message'].append(AIMessage(content=response.content))
    print("\nAI: "+response.content)
    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge('process',END)
agent = graph.compile()

conversation_history = []
user_input = input("Enter: ")
while user_input != 'exit':
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"message":conversation_history})
    # print(result['message'])
    conversation_history = result['message']
    user_input = input("Enter: ")

with open('config.txt','w',encoding='utf-8') as file:
    file.write("Your conversation Log:\m")

    for message in conversation_history:
        if(isinstance(message, HumanMessage)):
            file.write(f"You {message.content}\n")
        
        elif(isinstance(message, AIMessage)):
            file.write(f"AI: {message.content}\n")
    
    file.write("End conversation")
print("cuoc hoi thoai da luu")