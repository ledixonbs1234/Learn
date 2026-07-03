# skills_library.py
import os
import openpyxl
import pandas as pd
from typing import Dict, Any, Callable
from config import sanitize_and_resolve_path

class SkillManager:
    """Quản lý việc lưu trữ, tra cứu tài liệu và thực thi các kỹ năng động."""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.skills: Dict[str, Dict[str, Any]] = {}
        self._register_default_skills()

    def register_skill(self, name: str, description: str, parameters: Dict[str, Any], usage_example: str, handler: Callable):
        """Đăng ký một kỹ năng mới vào thư viện."""
        self.skills[name] = {
            "metadata": {
                "name": name,
                "description": description,
                "parameters": parameters,
                "usage_example": usage_example
            },
            "handler": handler
        }

    def get_all_skills_summary(self) -> str:
        """Trả về danh sách tóm tắt tất cả kỹ năng hiện có để Agent biết mình có những khả năng gì."""
        summary = []
        for name, data in self.skills.items():
            summary.append(f"- **{name}**: {data['metadata']['description']}")
        return "\n".join(summary)

    def get_skill_detail(self, name: str) -> str:
        """Trả về hướng dẫn chi tiết của một kỹ năng cụ thể cho Agent đọc."""
        if name not in self.skills:
            return f"Lỗi: Không tìm thấy kỹ năng '{name}'."
        meta = self.skills[name]["metadata"]
        return (
            f"=== CHI TIẾT KỸ NĂNG: {meta['name']} ===\n"
            f"Mô tả: {meta['description']}\n"
            f"Tham số yêu cầu: {meta['parameters']}\n"
            f"Ví dụ sử dụng: {meta['usage_example']}\n"
        )

    def execute(self, name: str, params: Dict[str, Any]) -> Any:
        """Thực thi logic xử lý của kỹ năng."""
        if name not in self.skills:
            raise ValueError(f"Không tìm thấy kỹ năng '{name}'")
        return self.skills[name]["handler"](self.workspace_path, params)

    def _register_default_skills(self):
        # 1. KỸ NĂNG: Xem thông tin cấu trúc tệp Excel
        self.register_skill(
            name="excel_inspect_structure",
            description="Đọc danh sách các sheet và tiêu đề cột của một tệp Excel.",
            parameters={
                "file_path": {"type": "string", "description": "Đường dẫn tương đối của tệp excel.", "required": True}
            },
            usage_example="excel_inspect_structure(file_path='data/report.xlsx')",
            handler=self._handle_inspect_structure
        )
        
        # 2. KỸ NĂNG: Chỉnh sửa giá trị ô Excel
        self.register_skill(
            name="excel_write_cell",
            description="Chỉnh sửa hoặc ghi đè một ô cụ thể trong Excel bằng openpyxl để bảo toàn định dạng.",
            parameters={
                "file_path": {"type": "string", "description": "Đường dẫn tệp.", "required": True},
                "sheet_name": {"type": "string", "description": "Tên sheet.", "required": True},
                "coordinate": {"type": "string", "description": "Tọa độ ô, ví dụ: 'C5'.", "required": True},
                "value": {"type": "any", "description": "Giá trị cần lưu.", "required": True}
            },
            usage_example="excel_write_cell(file_path='report.xlsx', sheet_name='Sheet1', coordinate='B2', value=150.5)",
            handler=self._handle_write_cell
        )

    # --- Các hàm xử lý nghiệp vụ thực tế dưới dạng Handler ---
    def _handle_inspect_structure(self, workspace: str, params: Dict[str, Any]) -> str:
        file_path = params.get("file_path")
        safe_path = sanitize_and_resolve_path(workspace, file_path)
        xl = pd.ExcelFile(safe_path)
        result = {}
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, nrows=2)
            result[sheet] = {"columns": list(df.columns)}
        return f"Cấu trúc tệp Excel:\n{result}"

    def _handle_write_cell(self, workspace: str, params: Dict[str, Any]) -> str:
        file_path = params.get("file_path")
        sheet_name = params.get("sheet_name")
        coordinate = params.get("coordinate")
        value = params.get("value")
        
        safe_path = sanitize_and_resolve_path(workspace, file_path)
        wb = openpyxl.load_workbook(safe_path)
        ws = wb[sheet_name]
        ws[coordinate] = value
        wb.save(safe_path)
        wb.close()
        return f"Đã ghi thành công giá trị '{value}' vào ô {coordinate} trên sheet '{sheet_name}'"