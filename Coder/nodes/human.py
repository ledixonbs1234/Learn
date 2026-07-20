# oder/nodes/human.py
import json
from typing import Dict, Any
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.types import interrupt

from state import AgentState

def human_interaction_gate_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    target_tool_msg = next((msg for msg in reversed(messages) if isinstance(msg, ToolMessage) and msg.name == "ask_questions_if_underspecified"), None)
    if not target_tool_msg:
        return {}
        
    try:
        data = json.loads(target_tool_msg.content)
        if not isinstance(data, dict) or data.get("status") != "requires_human_response":
            return {}
        payload = data.get("payload", {})
    except Exception:
        return {}

    user_input = interrupt(payload)
    feedback_content = f"### [Phản hồi của người dùng cho các câu hỏi]:\n{json.dumps(user_input, ensure_ascii=False)}"
    feedback_message = HumanMessage(content=feedback_content, name="human_interaction_feedback")
    return {"messages": [feedback_message]}