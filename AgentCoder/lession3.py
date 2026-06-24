import os
import re
from typing import Annotated, Dict, List, TypedDict, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class ProjectState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
    spec :str
    files_to_create:List[str]
    files:Dict[str,str]
    current_file_index:int
    
model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.3
)

def extract_file_content(text:str,filename:str)->str:
    ext = filename.split('.')[-1]
    if ext=='py':
        match = re.search(r"```python\n(.*?)\n```",text,re.DOTALL)
    elif ext == 'md':
        match = re.search(r"```markdown\n(.*?)\n```",text,re.DOTALL)
        if not match:
            match = re.search(r"```\n(.*?)\n```",text,re.DOTALL)

    else:
        match = re.search(r"```(.*?)\n(.*?)\n```", text, re.DOTALL)
        if match and match.group(2):
            return match.group(2)
    
    if match:
        return match.group(1)
    return text.replace("```python", "").replace("```", "").strip()

# Node 1: Architect - Thiết kế và lên danh sách tệp tin
def architect_node(state:ProjectState):
    print("\n--- 🏗️ [NODE: ARCHITECT] ĐANG LÊN THIẾT KẾ DỰ ÁN ---")
    messages = list(state["messages"])

    prompt = (
        "Bạn là một Solution Architect chuyên nghiệp.\n"
        "Hãy thiết kế cấu trúc dự án đáp ứng yêu cầu của người dùng.\n"
        "Yêu cầu bắt buộc trong phản hồi:\n"
        "1. Liệt kê danh sách các tệp tin cần tạo theo định dạng dòng: 'FILES: file1, file2, file3'\n"
        "2. Giải thích kiến trúc tổng quan của dự án và nhiệm vụ của từng tệp tin.\n"
        "Lưu ý: Chỉ thiết kế tối đa 2-3 tệp tin cần thiết nhất để tránh quá tải."
    )
    
    response = model.invoke([SystemMessage(content=prompt)]+messages)
    spec = response.content

    
    files_to_create = []
    match = re.search(r"FILE:\s(.*)",spec,re.IGNORECASE)
    if match:
        files_raw = match.group(1).split(',')
        files_to_create = [f.strip() for f in files_raw if f.strip()]
    else:
        files_to_create = ["main.py","utils.py",'README.md']
    print(f"👉 Đề xuất danh sách tệp tin: {files_to_create}")
    print("👉 Bản thiết kế sơ bộ:\n", spec)
    
    return {
        "messages":[response],
        "spec":spec,
        "files_to_create":files_to_create,
        "current_file_index":0,
        "files":{}
    }

def human_review_node(state:ProjectState):
    print("\n--- 👁️ [NODE: HUMAN REVIEW] CON NGƯỜI ĐANG KIỂM DUYỆT KIẾN TRÚC ---")
    pass

def code_generator_node(state:ProjectState):
    files_to_create = state['files_to_create']
    current_index = state['current_file_index']
    files = state.get('files',{})
    spec = state["spec"]

    current_file = files_to_create[current_index]
    print(f"\n--- 💻 [NODE: CODER] ĐANG VIẾT CODE CHO TỆP: {current_file} ({current_index + 1}/{len(files_to_create)}) ---")
    
    prompt = (
        f"Dựa trên bản thiết kế kiến trúc sau đây:\n{spec}\n\n"
        f"Hãy viết mã nguồn hoàn chỉnh, sạch sẽ và thực thi được cho tệp tin: '{current_file}'.\n"
        f"Lưu ý: Chỉ trả về nội dung mã nguồn bên trong khối code Markdown tương ứng (ví dụ: ```python ... ```)."
    )
    
    response = model.invoke([HumanMessage(content=prompt)])
    file_content = extract_file_content(response.content,current_file)
    files[current_file] = file_content
    return {
        "messages":[response],
        "files":files,
        "current_file_index":current_index+1
    }
    
def file_writer_node(state:ProjectState):
    print("\n--- 📁 [NODE: FILE WRITER] ĐANG GHI CÁC TỆP TIN XUỐNG ĐĨA ---")
    files = state['files']
    project_dir = './output_project'

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    
    for file_path, content in files.items():
        safe_name = os.path.basename(file_path)
        full_path = os.path.join(project_dir,safe_name)

        with open(full_path,"w",encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Đã tạo tệp tin thành công: {full_path}")
        
    return {} 

def should_generate_more_files(state:ProjectState):
    current_index = state["current_file_index"]
    files_to_create = state["files_to_create"]

    if current_index < len(files_to_create):
        print("-> Vòng lặp: Tiếp tục sinh tệp tin tiếp theo...")
        return "code_generator"
    
    return 'file_writer'

builder = StateGraph(ProjectState)
builder.add_node("architect",architect_node)
builder.add_node("human_review",human_review_node)
builder.add_node("code_generator",code_generator_node)
builder.add_node("file_writer",file_writer_node)

builder.add_edge(START,'architect')
builder.add_edge('architect','human_review')
builder.add_edge('human_review',"code_generator")
builder.add_conditional_edges("code_generator",
                              should_generate_more_files,
                              {
                                  "code_generator":"code_generator",
                                  "file_writer":"file_writer"
                              })
    
builder.add_edge("file_writer",END)
memory = MemorySaver()
app = builder.compile(checkpointer=memory,interrupt_before=['human_review'])

def main():
    config = {"configurable": {"thread_id": "project_build_session"}}
    user_query = "Tạo một công cụ dòng lệnh (CLI) bằng Python để quản lý danh sách công việc (Todo List) có chức năng thêm, xem và xóa."
    print(f"Yêu cầu dự án: '{user_query}'\n")
    print("[Hệ thống] Đang chạy bước Thiết kế kiến trúc...")
    for event in app.stream({"messages": [HumanMessage(content=user_query)]}, config, stream_mode="values"):
        pass
    state_snapshot = app.get_state(config)
    spec = state_snapshot.values.get('spec')
    proposed_files = state_snapshot.values.get('files_to_create',[])
    print("\n==================================================")
    print("🏗️ KIẾN TRÚC DO AI ĐỀ XUẤT:")
    print(spec)
    print("==================================================")
    print(f"Danh sách tệp tin AI định tạo: {proposed_files}")
    print("\nBạn muốn làm gì?")
    print("1. Đồng ý và cho phép tiến hành viết code.")
    print("2. Chỉnh sửa danh sách tệp tin trước khi viết code.")
    choice = input("Nhập lựa chọn của bạn (1 hoặc 2): ").strip()
    
    if choice == "2":
        print("\nVí dụ bạn muốn bỏ bớt hoặc thêm tệp tin mới.")
        user_input_files = input("Nhập lại danh sách tệp tin mới (ngăn cách bởi dấu phẩy):\n> ") 
        new_files_list = [f.strip() for f in user_input_files.split(',') if f.trip()]
        app.update_state(
            config,
            {"files_to_create":new_files_list},
            as_node="architect"
        )
        print(f"\n[Hệ thống] Đã cập nhật danh sách tệp tin mới: {new_files_list}")
    else:
        print("\n[Hệ thống] Đã phê duyệt cấu trúc đề xuất từ AI.")
        
    print("\n[Hệ thống] Tiến hành sinh code cho từng tệp tin...")
    for event in app.stream(None, config, stream_mode="values"):
        pass
    print("\n[Hệ thống] Hoàn tất dự án! Hãy kiểm tra thư mục './output_project' trên máy tính của bạn.")
    
if __name__ == "__main__":
    main()