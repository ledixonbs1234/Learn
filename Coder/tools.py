# tools.py
import json
import subprocess
import re  # <--- THÊM IMPORT THƯ VIỆN RE Ở ĐẦU FILE
from pathlib import Path
from typing import List, Union, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from config import GitIgnoreMatcher, sanitize_and_resolve_path

# ==========================================
# SCHEMAS PHỤC VỤ TOOL CALLING
# ==========================================
class ReadFilesSchema(BaseModel):
    file_paths: Union[str, List[str]] = Field(description="Đường dẫn tương đối hoặc danh sách đường dẫn tương đối của các tệp tin.")

class WriteFileSchema(BaseModel):
    file_path: str = Field(description="Đường dẫn tương đối của tệp tin trong workspace cần ghi hoặc cập nhật.")
    content: str = Field(description="Toàn bộ nội dung tệp tin chi tiết cần lưu xuống đĩa.")

class ApplyPatchSchema(BaseModel):  # <--- THÊM SCHEMA CHO BẢN VÁ MỚI
    file_path: str = Field(description="Đường dẫn tương đối của tệp tin trong workspace cần sửa đổi.")
    patch_block: str = Field(
        description="Khối bản vá Tìm kiếm & Thay thế (Search-and-Replace) bắt buộc viết theo định dạng:\n"
                    "<<<<<<< SEARCH\n"
                    "[Mã cũ cần thay thế khớp chính xác]\n"
                    "=======\n"
                    "[Mã mới cần đổi]\n"
                    ">>>>>>> REPLACE"
    )

class ListDirSchema(BaseModel):
    sub_dir: str = Field(default=".", description="Đường dẫn tương đối của thư mục cần xem.")

class RunTerminalSchema(BaseModel):
    command: str = Field(description="Lệnh terminal hệ điều hành cần thực thi trực tiếp tại thư mục gốc của workspace.")

# ==========================================
# CÁC LỚP CÔNG CỤ CHUẨN HOÁ (BASE TOOL)
# ==========================================
class ReadFilesTool(BaseTool):
    name: str = "read_files"
    description: str = "Đọc nội dung của một hoặc nhiều tệp tin trong workspace cùng lúc."
    args_schema: Type[BaseModel] = ReadFilesSchema
    workspace_path: str

    def _run(self, file_paths: Union[str, List[str]]) -> str:
        tools_mgr = WorkspaceTools(self.workspace_path)
        return tools_mgr.read_files(file_paths)


class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "Ghi mới hoặc cập nhật nội dung chi tiết của một tệp tin trong workspace."
    args_schema: Type[BaseModel] = WriteFileSchema
    workspace_path: str

    def _run(self, file_path: str, content: str) -> str:
        tools_mgr = WorkspaceTools(self.workspace_path)
        return tools_mgr.write_file(file_path, content)


class ApplyPatchTool(BaseTool):  # <--- THÊM CLASS LỚP CÔNG CỤ APY PATCH TOOL
    name: str = "apply_search_replace_patch"
    description: str = "Áp dụng bản vá sửa đổi cục bộ tối giản vào một tệp tin thông qua khối Tìm kiếm & Thay thế (Search-and-Replace)."
    args_schema: Type[BaseModel] = ApplyPatchSchema
    workspace_path: str

    def _run(self, file_path: str, patch_block: str) -> str:
        tools_mgr = WorkspaceTools(self.workspace_path)
        return tools_mgr.apply_search_replace_patch(file_path, patch_block)


class ListDirectoryTool(BaseTool):
    name: str = "list_directory"
    description: str = "Liệt kê các tệp và thư mục con trong thư mục chỉ định đệ quy."
    args_schema: Type[BaseModel] = ListDirSchema
    workspace_path: str

    def _run(self, sub_dir: str = ".") -> str:
        tools_mgr = WorkspaceTools(self.workspace_path)
        return tools_mgr.list_directory(sub_dir)


class RunTerminalTool(BaseTool):
    name: str = "run_terminal_command"
    description: str = "Thực thi một lệnh terminal hệ điều hành trực tiếp trong thư mục gốc của workspace."
    args_schema: Type[BaseModel] = RunTerminalSchema
    workspace_path: str

    def _run(self, command: str) -> str:
        tools_mgr = WorkspaceTools(self.workspace_path)
        return tools_mgr.run_terminal(command)

# ==========================================
# CÁC LỚP TIỆN ÍCH NGHIỆP VỤ
# ==========================================
class GitManager:
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path).resolve()
        
    def _run_cmd(self, args: list, ignore_error: bool = False) -> str:
        try:
            res = subprocess.run(args, cwd=str(self.workspace), capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except subprocess.CalledProcessError as e:
            if ignore_error:
                return f"ERROR: {e.stderr.strip()}"
            raise RuntimeError(f"Lỗi thực thi lệnh Git {' '.join(args)}: {e.stderr.strip()}")
        except FileNotFoundError:
            if ignore_error:
                return "ERROR: Lệnh Git không khả dụng hoặc chưa được cài đặt."
            raise RuntimeError("Lỗi: Không tìm thấy hệ thống Git trong biến môi trường PATH.")

    def init_and_prepare_branch(self) -> str:
        git_dir = self.workspace / ".git"
        if not git_dir.exists():
            # Thay vì tự ý chạy "git init", chúng ta ném ra lỗi để cảnh báo
            # (Node xử lý phía trên sẽ bắt điều kiện này trước khi gọi hàm)
            raise RuntimeError("Thư mục hiện tại không phải là một Git repository.")
            
        branch_name = "ai-development"
        branches_str = self._run_cmd(["git", "branch"], ignore_error=True)
        
        if branch_name in branches_str:
            self._run_cmd(["git", "checkout", branch_name])
        else:
            self._run_cmd(["git", "checkout", "-b", branch_name])
            
        return branch_name

    def commit_changes(self, message: str):
        self._run_cmd(["git", "add", "."])
        status = self._run_cmd(["git", "status", "--porcelain"], ignore_error=True)
        if status and not status.startswith("ERROR"):
            self._run_cmd(["git", "commit", "-m", message])


class WorkspaceTools:
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path).resolve()

    def _normalize_file_paths(self, file_paths: Union[str, List[str]]) -> List[str]:
        """Chuẩn hóa mọi dạng đầu vào từ LLM (chuỗi JSON, chuỗi thô, list) về List[str] sạch sẽ."""
        if not file_paths:
            return []
            
        # 1. Nếu đã là một List thực sự
        if isinstance(file_paths, list):
            cleaned = []
            for p in file_paths:
                if isinstance(p, str):
                    # Gọi đệ quy đề phòng trường hợp phần tử bên trong list vẫn bị bọc chuỗi JSON
                    cleaned.extend(self._normalize_file_paths(p))
            return cleaned
            
        # 2. Nếu đầu vào là một String
        if isinstance(file_paths, str):
            raw_str = file_paths.strip()
            
            # Kiểm tra nếu chuỗi trông giống như một JSON Array (bắt đầu bằng [ và kết thúc bằng ])
            if raw_str.startswith('[') and raw_str.endswith(']'):
                try:
                    # Thử giải mã JSON
                    parsed = json.loads(raw_str)
                    if isinstance(parsed, list):
                        return [str(p).strip() for p in parsed if p]
                except json.JSONDecodeError:
                    pass
                
                # Nếu không phải JSON hợp lệ (ví dụ: [lib/main.dart, lib/routes.dart] không có dấu nháy)
                # Tiến hành cắt bỏ cặp ngoặc [] và tách thủ công bằng dấu phẩy
                inner = raw_str[1:-1].strip()
                if inner:
                    parts = []
                    for part in inner.split(','):
                        part_clean = part.strip().strip('"').strip("'").strip()
                        if part_clean:
                            parts.append(part_clean)
                    return parts
            
            # Kiểm tra nếu chuỗi chứa dấu phẩy phân cách mà không phải cấu trúc JSON
            if ',' in raw_str and not raw_str.startswith('{'):
                parts = []
                for part in raw_str.split(','):
                    part_clean = part.strip().strip('"').strip("'").strip()
                    if part_clean:
                        parts.append(part_clean)
                return parts
            
            # Xử lý trường hợp chuỗi đơn bị bọc nháy kép hoặc nháy đơn dư thừa từ LLM
            cleaned_path = raw_str.strip('"').strip("'").strip()
            if cleaned_path:
                return [cleaned_path]
                
        return []

    def read_files(self, file_paths: Union[str, List[str]]) -> str:
        # Sử dụng hàm chuẩn hoá đầu vào mới được thiết kế
        paths = self._normalize_file_paths(file_paths)
        if not paths:
            return "Lỗi: Danh sách đường dẫn tệp tin trống hoặc không thể phân tích định dạng."
            
        results = []
        for path in paths:
            try:
                safe_path = sanitize_and_resolve_path(str(self.workspace), path, create_parent=False)
                if not safe_path.exists():
                    results.append(f"--- THẤT BẠI: '{path}' (Tệp không tồn tại) ---")
                    continue
                if safe_path.is_dir():
                    results.append(f"--- THẤT BẠI: '{path}' (Đường dẫn được chỉ định là thư mục) ---")
                    continue
                
                content = safe_path.read_text(encoding="utf-8")
                results.append(f"=== BẮT ĐẦU NỘI DUNG TỆP: {path} ===\n{content}\n=== KẾT THÚC NỘI DUNG TỆP: {path} ===")
            except Exception as e:
                results.append(f"--- THẤT BẠI: '{path}' (Lỗi đọc tệp: {str(e)}) ---")
                
        return "\n\n".join(results)

    def write_file(self, file_path: str, content: str) -> str:
        try:
            safe_path = sanitize_and_resolve_path(str(self.workspace), file_path, create_parent=True)
            safe_path.write_text(content, encoding="utf-8")
            return f"Đã ghi và lưu tệp thành công tại đường dẫn: '{file_path}'"
        except Exception as e:
            return f"Lỗi ghi tệp: {str(e)}"

    def apply_search_replace_patch(self, file_path: str, patch_block: str) -> str:  # <--- HÀM XỬ LÝ VẬT LÝ BẢN VÁ
        try:
            safe_path = sanitize_and_resolve_path(str(self.workspace), file_path, create_parent=False)
            if not safe_path.exists():
                return f"Lỗi: Không tìm thấy tệp tin '{file_path}' cần áp dụng bản vá."
                
            # Phân tích cú pháp khối Search-and-Replace
            pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
            match = re.search(pattern, patch_block, re.DOTALL)
            
            if not match:
                return (
                    "Lỗi: Bản vá không đúng định dạng. Bạn bắt buộc phải tuân thủ chính xác cấu trúc:\n"
                    "<<<<<<< SEARCH\n"
                    "[Mã nguồn cũ cần tìm chính xác]\n"
                    "=======\n"
                    "[Mã nguồn mới thay thế]\n"
                    ">>>>>>> REPLACE"
                )
                
            search_code = match.group(1)
            replace_code = match.group(2)
            
            original_content = safe_path.read_text(encoding="utf-8")
            
            # Đảm bảo khối SEARCH tồn tại khớp chính xác trong tệp
            if search_code not in original_content:
                return (
                    f"Lỗi: Không tìm thấy phân đoạn mã SEARCH được chỉ định trong tệp '{file_path}'.\n"
                    "Hãy chắc chắn rằng bạn đã copy chính xác từng khoảng trắng, thụt lề dòng (indentation), "
                    "và ký tự xuống dòng từ nội dung gốc của tệp."
                )
                
            # Thay thế cục bộ (Chỉ thay thế khớp đầu tiên để đảm bảo tính toàn vẹn)
            new_content = original_content.replace(search_code, replace_code, 1)
            safe_path.write_text(new_content, encoding="utf-8")
            
            return f"Thành công: Đã áp dụng bản vá Search-and-Replace cho tệp '{file_path}'."
            
        except Exception as e:
            return f"Lỗi xảy ra khi áp dụng bản vá cục bộ: {str(e)}"

    def list_directory(self, sub_dir: str = ".") -> str:
        try:
            safe_path = sanitize_and_resolve_path(str(self.workspace), sub_dir, create_parent=False)
            if not safe_path.exists():
                return f"Lỗi: Thư mục '{sub_dir}' không tồn tại."
            
            matcher = GitIgnoreMatcher(self.workspace)
            
            def traverse(current_path: Path, depth: int = 0, max_depth: int = 4) -> List[str]:
                if depth > max_depth:
                    return []
                
                results = []
                try:
                    items = sorted(list(current_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
                except Exception as e:
                    return [f"{'  ' * depth}[Lỗi truy cập: {str(e)}]"]
                
                for item in items:
                    if matcher.is_ignored(item):
                        continue
                    indent = "  " * depth
                    if item.is_dir():
                        results.append(f"{indent}📁 {item.name}/")
                        results.extend(traverse(item, depth + 1, max_depth))
                    else:
                        results.append(f"{indent}📄 {item.name}")
                return results

            tree_lines = traverse(safe_path, depth=0, max_depth=4)
            if not tree_lines:
                return f"Thư mục '{sub_dir}' trống hoặc toàn bộ các tệp tin bên trong đã bị loại bỏ theo cấu hình .gitignore."
                
            header = f"Cấu trúc thư mục tương đối của '{sub_dir}' (Đã lọc bỏ tệp tin .gitignore):\n"
            return header + "\n".join(tree_lines)
        except Exception as e:
            return f"Lỗi liệt kê thư mục: {str(e)}"

    def run_terminal(self, command: str, timeout: int = 60) -> str:
        try:
            res = subprocess.run(
                command,
                cwd=str(self.workspace),
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = []
            if res.stdout:
                output.append(f"[STDOUT]\n{res.stdout.strip()}")
            if res.stderr:
                output.append(f"[STDERR]\n{res.stderr.strip()}")
                
            status_msg = f"Lệnh kết thúc với exit code: {res.returncode}"
            if not output:
                return f"{status_msg} (Không có dữ liệu đầu ra)"
                
            return f"{status_msg}\n" + "\n\n".join(output)
            
        except subprocess.TimeoutExpired:
            return f"Lỗi: Lệnh bị buộc dừng do vượt quá thời gian chờ (timeout) {timeout} giây."
        except Exception as e:
            return f"Lỗi thực thi lệnh terminal: {str(e)}"
        
        
@tool
def get_current_working_directory() -> str:
    """Trả về đường dẫn thư mục làm việc hiện tại (Current Working Directory - CWD) của tiến trình đang chạy Agent."""
    return str(Path.cwd().resolve())


@tool
def check_path_exists(path_str: str) -> str:
    """Kiểm tra xem một đường dẫn cụ thể trên máy tính có tồn tại không và trả về đường dẫn tuyệt đối đã được chuẩn hóa (bao gồm cả việc dịch dấu ~)."""
    try:
        p = Path(path_str).expanduser().resolve()
        exists = p.exists()
        is_dir = p.is_dir() if exists else False
        return json.dumps({
            "exists": exists,
            "is_directory": is_dir,
            "resolved_absolute_path": str(p) if exists else None
        }, ensure_ascii=False)
    except Exception as e:
        return f"Lỗi kiểm tra đường dẫn: {str(e)}"


@tool
def find_project_root(start_path: str) -> str:
    """Tìm kiếm ngược lên trên (upwards) từ một đường dẫn bắt đầu để tìm kiếm các tệp tin đánh dấu gốc dự án như '.git', 'package.json', 'pyproject.toml', 'requirements.txt', 'THONGTIN.md'."""
    try:
        current = Path(start_path).expanduser().resolve()
        markers = [".git", "package.json", "pyproject.toml", "requirements.txt", "THONGTIN.md", "main.py"]
        
        # Quét ngược lên tối đa 5 cấp thư mục cha để tìm thư mục gốc thực sự của dự án
        for _ in range(5):
            for marker in markers:
                if (current / marker).exists():
                    return json.dumps({
                        "found": True,
                        "project_root": str(current),
                        "matched_marker": marker
                    }, ensure_ascii=False)
            if current.parent == current:
                break
            current = current.parent
            
        return json.dumps({"found": False, "message": "Không tìm thấy file đánh dấu dự án nào ở các thư mục cha."}, ensure_ascii=False)
    except Exception as e:
        return f"Lỗi tìm kiếm dự án gốc: {str(e)}"