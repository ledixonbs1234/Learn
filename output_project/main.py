#!/usr/bin/env python3
"""
Todo CLI - A simple command-line todo list manager
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    # Fallback if colorama is not installed
    class Fore:
        YELLOW = ""
        GREEN = ""
        RED = ""
        CYAN = ""
        RESET = ""
    
    class Style:
        BRIGHT = ""
        RESET_ALL = ""


TODO_FILE = "todos.json"


def load_todos() -> List[Dict]:
    """Load todos from JSON file"""
    if not os.path.exists(TODO_FILE):
        return []
    
    try:
        with open(TODO_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_todos(todos: List[Dict]) -> None:
    """Save todos to JSON file"""
    with open(TODO_FILE, 'w', encoding='utf-8') as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)


def get_next_id(todos: List[Dict]) -> int:
    """Get the next available ID"""
    if not todos:
        return 1
    return max(todo['id'] for todo in todos) + 1


def add_todo(task: str) -> None:
    """Add a new todo"""
    todos = load_todos()
    new_todo = {
        'id': get_next_id(todos),
        'task': task,
        'status': 'pending',
        'created_at': datetime.now().isoformat()
    }
    todos.append(new_todo)
    save_todos(todos)
    print(f"{Fore.GREEN}✓ Đã thêm công việc #{new_todo['id']}: {task}")


def view_todos(filter_status: Optional[str] = None) -> None:
    """View todos with optional status filter"""
    todos = load_todos()
    
    if not todos:
        print(f"{Fore.YELLOW}Không có công việc nào.")
        return
    
    # Filter todos based on status
    if filter_status == 'pending':
        filtered_todos = [t for t in todos if t['status'] == 'pending']
        title = "CÔNG VIỆC ĐANG LÀM"
    elif filter_status == 'completed':
        filtered_todos = [t for t in todos if t['status'] == 'completed']
        title = "CÔNG VIỆC ĐÃ HOÀN THÀNH"
    else:
        filtered_todos = todos
        title = "TẤT CẢ CÔNG VIỆC"
    
    if not filtered_todos:
        print(f"{Fore.YELLOW}Không có công việc nào trong danh mục này.")
        return
    
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{title:^60}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}\n")
    
    for todo in filtered_todos:
        status_symbol = "✓" if todo['status'] == 'completed' else "○"
        color = Fore.GREEN if todo['status'] == 'completed' else Fore.YELLOW
        
        # Format datetime
        created = datetime.fromisoformat(todo['created_at'])
        date_str = created.strftime("%Y-%m-%d %H:%M")
        
        print(f"{color}[{status_symbol}] #{todo['id']}: {todo['task']}")
        print(f"    {Fore.CYAN}Tạo lúc: {date_str}\n")


def delete_todo(todo_id: int) -> None:
    """Delete a todo by ID"""
    todos = load_todos()
    
    # Find and remove the todo
    original_len = len(todos)
    todos = [t for t in todos if t['id'] != todo_id]
    
    if len(todos) == original_len:
        print(f"{Fore.RED}✗ Không tìm thấy công việc #{todo_id}")
        return
    
    save_todos(todos)
    print(f"{Fore.GREEN}✓ Đã xóa công việc #{todo_id}")


def complete_todo(todo_id: int) -> None:
    """Mark a todo as completed"""
    todos = load_todos()
    
    # Find and update the todo
    found = False
    for todo in todos:
        if todo['id'] == todo_id:
            if todo['status'] == 'completed':
                print(f"{Fore.YELLOW}⚠ Công việc #{todo_id} đã được hoàn thành trước đó")
                return
            todo['status'] = 'completed'
            found = True
            break
    
    if not found:
        print(f"{Fore.RED}✗ Không tìm thấy công việc #{todo_id}")
        return
    
    save_todos(todos)
    print(f"{Fore.GREEN}✓ Đã đánh dấu hoàn thành công việc #{todo_id}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Todo CLI - Quản lý công việc đơn giản",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Lệnh có thể sử dụng')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Thêm công việc mới')
    add_parser.add_argument('task', type=str, help='Nội dung công việc')
    
    # View command
    view_parser = subparsers.add_parser('view', help='Xem danh sách công việc')
    view_group = view_parser.add_mutually_exclusive_group()
    view_group.add_argument('--all', action='store_true', help='Xem tất cả công việc')
    view_group.add_argument('--pending', action='store_true', help='Chỉ xem công việc đang làm')
    view_group.add_argument('--completed', action='store_true', help='Chỉ xem công việc đã hoàn thành')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Xóa công việc')
    delete_parser.add_argument('id', type=int, help='ID công việc cần xóa')
    
    # Complete command
    complete_parser = subparsers.add_parser('complete', help='Đánh dấu hoàn thành')
    complete_parser.add_argument('id', type=int, help='ID công việc cần hoàn thành')
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'add':
        add_todo(args.task)
    elif args.command == 'view':
        if args.pending:
            view_todos('pending')
        elif args.completed:
            view_todos('completed')
        else:
            view_todos('all')
    elif args.command == 'delete':
        delete_todo(args.id)
    elif args.command == 'complete':
        complete_todo(args.id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()