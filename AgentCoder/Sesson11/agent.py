import time
import uuid
from typing import TypedDict, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
class Candidate(TypedDict):
    id: str
    strategy_name: str
    code: str
    reasoning: str
    score: Optional[float]
    feedback: Optional[str]
    iteration: int

class MCTSState(TypedDict):
    problem: str                # Đề bài yêu cầu giải quyết
    candidates: List[Candidate] # Danh sách tất cả các phương án đã từng sinh ra
    iteration: int              # Số lần lặp hiện tại
    max_iterations: int         # Giới hạn số lần thử
    best_candidate: Optional[Candidate] # Phương án tốt nhất tìm được cho đến nay
    log: List[str]   


class StrategyProposal(BaseModel):
    strategy_name: str = Field(
        ..., 
        description="Tên của giải thuật đề xuất (ví dụ: Quy hoạch động, Đệ quy có nhớ, v.v.)"
    )
    code: str = Field(
        ..., 
        description="Mã nguồn Python hoàn chỉnh của hàm lcs(s1, s2). Phải định nghĩa hàm 'def lcs(s1: str, s2: str) -> int:'. Không import các thư viện ngoài ngoại trừ thư viện chuẩn Python."
    )
    reasoning: str = Field(
        ..., 
        description="Phân tích độ phức tạp thời gian/không gian và lý do chọn hướng đi này."
    )

class ProposeStrategiesTool(BaseModel):
    """Công cụ dùng để đề xuất chính xác 3 phương án giải thuật khác nhau cho bài toán."""
    proposals: List[StrategyProposal] = Field(
        ..., 
        description="Danh sách gồm 3 phương án giải thuật khác nhau để giải quyết bài toán."
    )

model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco", # type: ignore
    model="kiro",
    temperature=0.1
).bind_tools([ProposeStrategiesTool], tool_choice="ProposeStrategiesTool")



def generator_node(state: MCTSState) -> Dict[str, Any]:
    iteration = state.get("iteration", 1)
    problem = state["problem"]
    candidates = state.get('candidates',[])
    
    # Xây dựng ngữ cảnh từ lịch sử các phương án đã thử
    history_context = ""
    if candidates:
        history_context = "\nCác phương án đã thử nghiệm trước đó và kết quả:\n"
        for cand in state["candidates"]:
            history_context += f"- Chiến thuật: {cand['strategy_name']} (Điểm: {cand['score']}/100)\n"
            history_context += f"  Lỗi/Phản hồi: {cand['feedback']}\n"
            history_context += "---------------------------------------\n"

    system_prompt = (
        "Bạn là một chuyên gia thiết kế thuật toán cấp cao. Nhiệm vụ của bạn là đưa ra các phương pháp "
        "giải quyết bài toán được yêu cầu.\n"
        "Yêu cầu bắt buộc: Bạn phải đề xuất 3 phương án khác nhau hoàn toàn về mặt tư duy thuật toán "
        "(ví dụ: một phương án dùng Đệ quy, một dùng Quy hoạch động, một dùng Tham lam hoặc Hai con trỏ).\n"
        "Hãy sử dụng công cụ 'ProposeStrategiesTool' để gửi câu trả lời của bạn."
    )
    
    user_content = f"Bài toán cần giải quyết:\n{problem}\n"
    if history_context:
        user_content += f"\n{history_context}\nHãy phân tích các lỗi trên và đề xuất 3 phương án mới tốt hơn hoặc cải tiến sâu từ phương án có điểm cao nhất."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    response = model.invoke(messages)
    
    # Trích xuất dữ liệu từ Tool Call
    new_candidates = []
    tool_calls = response.tool_calls
    
    if tool_calls:
        # Lấy arguments từ tool call đầu tiên
        args = tool_calls[0]["args"]
        proposals = args.get("proposals", [])
        
        for prop in proposals:
            new_candidates.append({
                "id": str(uuid.uuid4()),
                "strategy_name": prop["strategy_name"],
                "code": prop["code"],
                "reasoning": prop["reasoning"],
                "score": None,
                "feedback": None,
                "iteration": iteration
            })
            
    log_msg = f"Vòng {iteration}: Đã sinh ra {len(new_candidates)} phương án mới."
    
    # Cập nhật danh sách candidates cũ bằng cách cộng thêm candidates mới
    updated_candidates = list(state.get("candidates", [])) + new_candidates
    
    return {
        "candidates": updated_candidates,
        "log": state.get("log", []) + [log_msg]
    }
    
def safe_run_and_evaluate(code_str: str) -> tuple[float, str]:
    """Biên dịch và thực thi thử nghiệm code trong một môi trường cô lập cục bộ."""
    local_vars = {}
    try:
        # Biên dịch code để kiểm tra lỗi cú pháp
        compiled = compile(code_str, "<string>", "exec")
        exec(compiled, {}, local_vars)
        
        # Kiểm tra sự tồn tại của hàm mục tiêu 'lcs'
        if "lcs" not in local_vars:
            return 0.0, "Lỗi: Không tìm thấy hàm định nghĩa 'lcs(s1, s2)' trong mã nguồn."
        
        lcs_fn = local_vars["lcs"]
        
        # Định nghĩa các Test Cases từ dễ đến khó (bao gồm cả trường hợp chuỗi lớn để kiểm tra tối ưu hóa)
        test_cases = [
            (("abcde", "ace"), 3),
            (("abc", "def"), 0),
            (("AGGTAB", "GXTXAYB"), 4),
            (("", ""), 0),
            (("aaaa", "aa"), 2),
            (("abcdefghijklmn", "apbcpdpepfpgph"), 8) # Test hiệu năng / độ phức tạp thuật toán
        ]
        
        passed = 0
        details = []
        
        start_time = time.perf_counter()
        for idx, ((s1, s2), expected) in enumerate(test_cases):
            try:
                # Thiết lập timeout giả lập (ví dụ nếu chạy quá lâu sẽ bị ngắt)
                res = lcs_fn(s1, s2)
                if res == expected:
                    passed += 1
                    details.append(f"TC {idx+1}: PASS")
                else:
                    details.append(f"TC {idx+1}: FAIL (Nhận {res}, mong đợi {expected})")
            except Exception as e:
                details.append(f"TC {idx+1}: ERROR ({str(e)})")
                
        execution_time = time.perf_counter() - start_time
        
        # Cách tính điểm: 
        # - Đúng hoàn toàn: 100 điểm.
        # - Nếu đúng hết nhưng chạy quá chậm (> 0.5s cho bộ test nhỏ): trừ 20 điểm (phạt thuật toán đệ quy chưa tối ưu).
        score = (passed / len(test_cases)) * 100
        if score == 100.0 and execution_time > 0.5:
            score = 80.0
            feedback = f"Đúng toàn bộ kết quả nhưng thời gian thực thi quá chậm ({execution_time:.4f}s). Cần tối ưu bằng Quy hoạch động hoặc Khử đệ quy."
        else:
            feedback = f"Kết quả: {', '.join(details)}. Thời gian chạy: {execution_time:.6f}s."
            
        return score, feedback
        
    except Exception as e:
        return 0.0, f"Lỗi biên dịch hoặc lỗi thực thi nghiêm trọng: {str(e)}"

def evaluator_node(state: MCTSState) -> Dict[str, Any]:
    iteration = state.get("iteration",1)
    candidates = list(state["candidates"])
    best_candidate = state.get("best_candidate")
    
    # Chỉ đánh giá các candidate của iteration hiện tại (chưa có điểm)
    evaluated_logs = []
    for cand in candidates:
        if cand["iteration"] == iteration and cand["score"] is None:
            score, feedback = safe_run_and_evaluate(cand["code"])
            cand["score"] = score
            cand["feedback"] = feedback
            evaluated_logs.append(f"[{cand['strategy_name']}] -> {score} điểm")
            
            # Cập nhật Best Candidate toàn cục nếu điểm số cao hơn
            if best_candidate is None or score > best_candidate["score"]: # type: ignore
                best_candidate = cand

    log_msg = f"Vòng {iteration} Đánh giá: " + " | ".join(evaluated_logs)
    
    return {
        "candidates": candidates,
        "best_candidate": best_candidate,
        "log": state.get("log", []) + [log_msg]
    }
    
    
    
def should_continue(state: MCTSState) -> str:
    best_candidate = state.get("best_candidate")
    
    # Điều kiện dừng 1: Có ứng viên đạt điểm tối đa
    if best_candidate and best_candidate["score"] >= 100.0: # type: ignore
        return "finish"
        
    # Điều kiện dừng 2: Vượt quá số lần lặp tối đa
    if state["iteration"] >= state["max_iterations"]:
        return "finish"
        
    return "next_iteration"

# Hàm trung gian để cập nhật số vòng lặp
def increment_iteration_node(state: MCTSState) -> Dict[str, Any]:
    return {
        "iteration": state["iteration"] + 1,
        "log": state.get("log", []) + [f"--- Chuyển sang Vòng {state['iteration'] + 1} ---"]
    }
    
builder = StateGraph(MCTSState)

# Thêm các Node
builder.add_node("generator", generator_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("increment_iteration", increment_iteration_node)

# Thiết lập các Edge
builder.add_edge(START, "generator")
builder.add_edge("generator", "evaluator")

# Thiết lập Conditional Edge sau khi đánh giá xong
builder.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "finish": END,
        "next_iteration": "increment_iteration"
    }
)

builder.add_edge("increment_iteration", "generator")

# Tích hợp Checkpointer để hỗ trợ quản lý bộ nhớ lịch sử và Quay lui trạng thái (Rollback)
memory = MemorySaver()
app = builder.compile(checkpointer=memory)