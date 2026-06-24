import json
import os
from datetime import datetime
from typing import List, Dict, Optional


def load_todos(filepath: str = "todos.json") -> List[Dict]:
    """
    Đọc danh sách công việc từ tệp JSON.
    
    Args:
        filepath: Đường dẫn đến tệp JSON
        
    Returns:
        Danh sách các công việc
    """
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def save_todos(todos: List[Dict], filepath: str = "todos.json") -> None:
    """
    Lưu danh sách công việc vào tệp JSON.
    
    Args:
        todos: Danh sách các công việc
        filepath: Đường dẫn đến tệp JSON
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)


def get_next_id(todos: List[Dict]) -> int:
    """
    Lấy ID tiếp theo cho công việc mới.
    
    Args:
        todos: Danh sách các công việc hiện tại
        
    Returns:
        ID tiếp theo
    """
    if not todos:
        return 1
    return max(todo['id'] for todo in todos) + 1


def find_todo_by_id(todos: List[Dict], todo_id: int) -> Optional[Dict]:
    """
    Tìm công việc theo ID.
    
    Args:
        todos: Danh sách các công việc
        todo_id: ID cần tìm
        
    Returns:
        Công việc tìm thấy hoặc None
    """
    for todo in todos:
        if todo['id'] == todo_id:
            return todo
    return None


def format_datetime(dt_string: str) -> str:
    """
    Định dạng chuỗi datetime thành dạng dễ đọc.
    
    Args:
        dt_string: Chuỗi datetime ISO format
        
    Returns:
        Chuỗi datetime đã định dạng
    """
    try:
        dt = datetime.fromisoformat(dt_string)
        return dt.strftime("%d/%m/%Y %H:%M")
    except (ValueError, AttributeError):
        return dt_string


def filter_todos(todos: List[Dict], status: Optional[str] = None) -> List[Dict]:
    """
    Lọc danh sách công việc theo trạng thái.
    
    Args:
        todos: Danh sách các công việc
        status: Trạng thái cần lọc ('pending', 'completed', hoặc None cho tất cả)
        
    Returns:
        Danh sách công việc đã lọc
    """
    if status is None:
        return todos
    return [todo for todo in todos if todo['status'] == status]