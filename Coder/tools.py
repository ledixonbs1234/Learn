# tools.py
import json
import subprocess
import re
from pathlib import Path
from typing import List, Union, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from config import GitIgnoreMatcher, sanitize_and_resolve_path

# ==========================================
# CÁC CÔNG CỤ DÒ TÌM HỆ THỐNG THỰC TẾ (SENSING TOOLS) [1.2.3]
# ==========================================
@tool
def get_current_working_directory() -> str:
    """Trả về đường dẫn thư mục làm việc hiện tại (Current Working Directory - CWD) của tiến trình đang chạy Agent."""
    return str(Path.cwd().resolve())


@tool
def check_path_exists(path_str: str) -> str:
    """Kiểm tra xem một đường dẫn cụ thể trên máy tính có tồn tại vật lý không và trả về đường dẫn tuyệt đối đã được chuẩn hóa (bao gồm cả việc dịch dấu ~)."""
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
        
        # Quét ngược lên tối đa 5 cấp thư mục cha
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

# ==========================================
# SCHEMAS PHỤC VỤ TOOL CALLING
# ==========================================
class ReadFilesSchema(BaseModel):
    file_paths: Union[str, List[str]] = Field(description="Đường dẫn tương đối hoặc danh sách đường dẫn tương đối của các tệp tin.")

class WriteFileSchema(BaseModel):
    file_path: str = Field(description="Đường dẫn tương đối của tệp tin trong workspace cần ghi hoặc cập nhật.")
    content: str = Field(description="Toàn bộ nội dung tệp tin chi tiết cần lưu xuống đĩa.")

class ApplyPatchSchema(BaseModel):
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

class ReadFileLinesSchema(BaseModel):
    file_path: str = Field(description="Đường dẫn tương đối của tệp tin trong workspace.")
    start_line: int = Field(description="Dòng bắt đầu đọc (đánh chỉ số từ 1).")
    end_line: int = Field(description="Dòng kết thúc đọc (bao gồm cả dòng này).")

class ReadFileLinesTool(BaseTool):
    name: str = "read_file_lines"
    description: str = (
        "Đọc một phân đoạn cụ thể của tệp tin trong phạm vi dòng [start_line] đến [end_line]. "
        "Giúp tập trung phân tích đoạn code cần thiết và tránh làm tràn bộ nhớ ngữ cảnh của mô hình."
    )
    args_schema: Type[BaseModel] = ReadFileLinesSchema
    workspace_path: str

    def _run(self, file_path: str, start_line: int, end_line: int) -> str:
        try:
            # Giải quyết đường dẫn tuyệt đối an toàn
            safe_path = sanitize_and_resolve_path(self.workspace_path, file_path, create_parent=False)
            if not safe_path.exists():
                return f"Lỗi: Không tìm thấy tệp '{file_path}'"
            
            if safe_path.is_dir():
                return f"Lỗi: '{file_path}' là một thư mục, vui lòng truyền đường dẫn tệp tin cụ thể."
            
            # Đọc toàn bộ dòng để xử lý cắt lát (slicing)
            content = safe_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            total_lines = len(lines)
            
            # Phòng thủ giới hạn đầu vào của Agent (bảo vệ biên)
            start = max(1, start_line)
            end = min(total_lines, end_line)
            
            if start > total_lines or start > end:
                return f"Lỗi: Phạm vi dòng yêu cầu ({start_line} - {end_line}) vượt quá giới hạn của tệp (Tệp hiện có {total_lines} dòng)."
            
            # Lấy các dòng được yêu cầu (mảng Python tính từ chỉ số 0)
            selected_lines = lines[start - 1 : end]
            
            # Định dạng đầu ra đi kèm số dòng hiển thị rõ ràng
            formatted_lines = []
            for idx, line in enumerate(selected_lines, start=start):
                formatted_lines.append(f"{idx:04d} | {line}")
                
            header = f"=== NỘI DUNG TỆP KHOANH VÙNG: {file_path} (Dòng {start} đến {end} trên tổng số {total_lines} dòng) ===\n"
            return header + "\n".join(formatted_lines)
            
        except Exception as e:
            return f"Lỗi xảy ra khi đọc phân đoạn dòng của tệp tin: {str(e)}"
        
        
class UniversalSearchSchema(BaseModel):
    file_path: str = Field(description="Đường dẫn tương đối của tệp tin nguồn trong workspace cần quét cấu trúc ký hiệu.")

class UniversalSymbolSearchTool(BaseTool):
    name: str = "search_symbols_universal"
    description: str = (
        "Quét tệp tin nguồn để trích xuất danh sách các định nghĩa Class, Hàm, Phương thức, Struct... "
        "Đi kèm với KHOẢNG DÒNG BẮT ĐẦU VÀ KẾT THÚC chính xác tuyệt đối của từng khối mã. "
        "Hỗ trợ tất cả các ngôn ngữ phổ biến (Python, Dart, TS, JS, Go, Rust, C++, Java, C#, Swift, Kotlin)."
    )
    args_schema: Type[BaseModel] = UniversalSearchSchema
    workspace_path: str

    def _clean_block_comments(self, content: str) -> str:
        """
        Xóa sạch comment khối /* ... */ đa dòng nhưng bảo toàn tuyệt đối số dòng 
        bằng cách đếm số dòng bị xóa và bù lại bằng bấy nhiêu ký tự xuống dòng.
        """
        def replacer(match):
            newlines_count = match.group(0).count("\n")
            return "\n" * newlines_count
            
        return re.sub(r"/\*.*?\*/", replacer, content, flags=re.DOTALL)

    def _strip_strings_and_line_comments(self, line: str, ext: str) -> str:
        """
        Xóa sạch comment một dòng và mọi chuỗi ký tự nằm trong ngoặc đơn/kép/template literal 
        để loại bỏ hoàn toàn các dấu ngoặc nhọn { } gây nhiễu trong chuỗi.
        """
        # 1. Xóa comment một dòng tùy theo ngôn ngữ
        if ext in [".py", ".yaml", ".yml"]:
            clean_line = re.sub(r"#.*$", "", line)
        else:
            clean_line = re.sub(r"//.*$", "", line)

        # 2. Xóa các chuỗi ký tự ngoặc kép "..." (bao gồm cả xử lý ký tự escape \")
        clean_line = re.sub(r'"(?:[^"\\]|\\.)*"', '""', clean_line)
        
        # 3. Xóa các chuỗi ký tự ngoặc đơn '...'
        clean_line = re.sub(r"'(?:[^'\\]|\\.)*'", "''", clean_line)
        
        # 4. Xóa chuỗi Template Literal `...` (cho JS, TS, Dart)
        clean_line = re.sub(r"`(?:[^`\\]|\\.)*`", "``", clean_line)
        
        return clean_line

    def _find_brace_block_end(self, lines: list, start_idx: int, ext: str) -> int:
        """Thuật toán Brace-Matching nâng cao: Khử chuỗi và comment trước khi đếm ngoặc nhọn."""
        brace_count = 0
        found_first_brace = False
        end_idx = start_idx
        
        # Phòng thủ: Kiểm tra xem đây có phải là Arrow Function một dòng không (JS/TS/Dart)
        first_line_clean = self._strip_strings_and_line_comments(lines[start_idx], ext)
        if "=>" in first_line_clean and first_line_clean.endswith(";"):
            return start_idx + 1 # Hàm một dòng, kết thúc ngay tại dòng bắt đầu [5]

        for idx in range(start_idx, len(lines)):
            line = lines[idx]
            # Làm sạch chuỗi và comment trên dòng hiện tại trước khi đếm ngoặc
            clean_line = self._strip_strings_and_line_comments(line, ext)
            
            for char in clean_line:
                if char == '{':
                    if not found_first_brace:
                        found_first_brace = True
                    brace_count += 1
                elif char == '}':
                    if found_first_brace:
                        brace_count -= 1
                        if brace_count == 0:
                            return idx + 1 # Tìm thấy ngoặc đóng khớp hoàn toàn
            
            if found_first_brace and brace_count == 0:
                return idx + 1
            
            end_idx = idx
            
        return end_idx + 1 # Fallback về cuối file nếu gặp file lỗi cú pháp bị khuyết ngoặc đóng

    def _find_python_block_end(self, lines: list, start_idx: int) -> int:
        """Thuật toán Indentation: Nhận diện ranh giới thụt lề của Python."""
        start_line = lines[start_idx]
        start_indent = len(start_line) - len(start_line.lstrip())
        
        end_idx = start_idx
        for idx in range(start_idx + 1, len(lines)):
            line = lines[idx]
            clean_line = line.strip()
            
            if not clean_line or clean_line.startswith('#') or clean_line.startswith('"""') or clean_line.startswith("'''"):
                continue
                
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= start_indent:
                return end_idx + 1
                
            end_idx = idx
            
        return end_idx + 1

    def _run(self, file_path: str) -> str:
        try:
            safe_path = sanitize_and_resolve_path(self.workspace_path, file_path, create_parent=False)
            if not safe_path.exists():
                return f"Thất bại: Không tìm thấy tệp tin '{file_path}'"
            
            raw_content = safe_path.read_text(encoding="utf-8")
            ext = safe_path.suffix.lower()
            
            # 1. Tiền xử lý: Xóa sạch block comments nhưng vẫn giữ nguyên vị trí dòng
            clean_content = self._clean_block_comments(raw_content)
            lines = clean_content.splitlines()
            
            # Bản đồ mẫu Regex hỗ trợ toàn diện các ngôn ngữ lập trình chính
            patterns = {
                ".py": [
                    r"^\s*(class\s+[a-zA-Z0-9_]+)",
                    r"^\s*(def\s+[a-zA-Z0-9_]+)"
                ],
               ".dart": [
                    r"\b(class|mixin|extension)\s+[a-zA-Z0-9_<>]+",
                    # Mẫu A: Có kiểu trả về đứng trước (bắt buộc có ít nhất 1 từ + khoảng trắng trước tên hàm)
                    r"^\s*(?:async\s+)?(?:[\w<>]+[\s\n]+)+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*(?:async\s*)?\{?",
                    # Mẫu B: Constructor hoặc phương thức không có kiểu trả về, bắt buộc kết thúc bằng {
                    r"^\s*(?:async\s+)?([a-zA-Z0-9_]+)\s*\([^)]*\)\s*(?:async\s*)?\{\s*$",
                    # Mẫu C: Hàm rút gọn => bắt buộc có ký tự =>
                    r"^\s*(?:async\s+)?([a-zA-Z0-9_]+)\s*\([^)]*\)\s*=>"
                ],
                ".ts": [
                    r"\b(class|interface|type)\s+[a-zA-Z0-9_]+",
                    r"\b(function\s+[a-zA-Z0-9_]+)",
                    r"\b(const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>"
                ],
                ".tsx": [
                    r"\b(class|interface)\s+[a-zA-Z0-9_]+",
                    r"\b(function\s+[a-zA-Z0-9_]+)",
                    r"\b(const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>"
                ],
                ".js": [
                    r"\b(class\s+[a-zA-Z0-9_]+)",
                    r"\b(function\s+[a-zA-Z0-9_]+)",
                    r"\b(const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>"
                ],
                ".rs": [
                    r"\b(?:pub\s+)?(?:struct|enum|union|trait)\s+([a-zA-Z0-9_]+)",
                    r"\b(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z0-9_]+)",
                    r"\bimpl(?:\s*<.*>)?\s+([a-zA-Z0-9_<>]+)"
                ],
                ".go": [
                    r"\btype\s+([a-zA-Z0-9_]+)\s+(struct|interface)",
                    r"\bfunc\s+(?:\([^)]+\)\s+)?([a-zA-Z0-9_]+)\s*\("
                ],
                ".cpp": [
                    r"\b(class|struct|namespace)\s+[a-zA-Z0-9_]+",
                    # Mẫu A: Có kiểu trả về đứng trước (bắt buộc có ít nhất một kiểu trả về / modifier)
                    r"^\s*(?:[\w&*<>:]+\s+)+([a-zA-Z0-9_~]+::)?([a-zA-Z0-9_~]+)\s*\([^)]*\)\s*(?:const|override|noexcept)?\s*\{?",
                    # Mẫu B: Constructor/Destructor không có kiểu trả về, bắt buộc kết thúc bằng { (hoặc có Initializer List :)
                    r"^\s*(?:[a-zA-Z0-9_~]+::)?([a-zA-Z0-9_~]+)\s*\([^)]*\)\s*(?::\s*[a-zA-Z0-9_~]+\(.*?\))*\s*\{\s*$"
                ],      
                ".h": [
                    r"\b(class|struct|namespace)\s+[a-zA-Z0-9_]+",
                    r"^\s*#define\s+[a-zA-Z0-9_]+",
                    r"^\s*(?:[\w&*<>:]+\s+)*([a-zA-Z0-9_~]+::)?([a-zA-Z0-9_~]+)\s*\([^)]*\)\s*(?:const|override|noexcept)?\s*;"
                ],
                ".hpp": [
                    r"\b(class|struct|namespace)\s+[a-zA-Z0-9_]+",
                    r"^\s*(?:[\w&*<>:]+\s+)*([a-zA-Z0-9_~]+::)?([a-zA-Z0-9_~]+)\s*\([^)]*\)\s*(?:const|override|noexcept)?\s*\{?"
                ],
                # 🛠️ THÊM MỚI: HỖ TRỢ JAVA (.java)
                ".java": [
                    r"\b(?:public|protected|private|static|\s)+(?:class|interface|enum)\s+([a-zA-Z0-9_]+)",
                    r"^\s*(?:@\w+\s+)*(?:public|protected|private|static|final|synchronized|\s)+(?:[\w<>]+)\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{?"
                ],
                # 🛠️ THÊM MỚI: HỖ TRỢ C# (.cs)
                ".cs": [
                    r"\b(?:public|protected|private|internal|static|\s)+(?:class|interface|struct|enum)\s+([a-zA-Z0-9_]+)",
                    r"^\s*(?:public|protected|private|internal|static|virtual|override|async|partial|\s)+(?:[\w<>]+)\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*\{?"
                ],
                # 🛠️ THÊM MỚI: HỖ TRỢ SWIFT (.swift)
                ".swift": [
                    r"\b(?:public|internal|private|fileprivate|open|\s)*(?:class|struct|protocol|enum|extension)\s+([a-zA-Z0-9_]+)",
                    r"\b(?:public|internal|private|fileprivate|open|static|class|override|async|\s)*func\s+([a-zA-Z0-9_]+)\s*\("
                ],
                # 🛠️ THÊM MỚI: HỖ TRỢ KOTLIN (.kt)
                ".kt": [
                    r"\b(?:public|internal|private|protected|sealed|data|\s)*(?:class|interface|object)\s+([a-zA-Z0-9_]+)",
                    r"\b(?:public|internal|private|protected|override|actual|expect|inline|tailrec|\s)*fun\s+(?:\([^)]+\)\s*)?([a-zA-Z0-9_]+)\s*\("
                ]
            }

            selected_patterns = patterns.get(ext, [r"\b(class\s+\w+)", r"\b(function\s+\w+)"])
            matched_symbols = []
            
            for line_no, line in enumerate(lines, 1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                
                # Bỏ qua các comment thuần một dòng trước khi quét mẫu signature
                if clean_line.startswith("//") or clean_line.startswith("#") or clean_line.startswith("/*") or clean_line.startswith("*"):
                    continue
                
                for pattern in selected_patterns:
                    if re.search(pattern, line):
                        start_line = line_no
                        
                        if ext == ".py":
                            end_line = self._find_python_block_end(lines, line_no - 1)
                        else:
                            end_line = self._find_brace_block_end(lines, line_no - 1, ext)
                        
                        matched_symbols.append(
                            f"Dòng {start_line:03d} -> Dòng {end_line:03d}: {clean_line}"
                        )
                        break
            
            if not matched_symbols:
                return f"Thông báo: Đã quét tệp '{file_path}' nhưng không phát hiện cấu trúc đặc trưng."
            
            header = f"=== BẢN ĐỒ PHẠM VI KHỐI MÃ (Ngôn ngữ: {ext.upper()}): {file_path} ===\n"
            return header + "\n".join(matched_symbols)

        except Exception as e:
            return f"Lỗi phân tích phạm vi khối mã: {str(e)}"
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


class ApplyPatchTool(BaseTool):
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
        self.workspace = Path(workspace_path).expanduser().resolve()
        
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
        # GIẢI PHÁP: Không tự khởi tạo git mới nếu chưa có sẵn. Báo lỗi để chuyển sang chế độ không Git.
        if not git_dir.exists():
            raise RuntimeError("Thư mục hiện tại chưa được khởi tạo Git repository.")
            
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
        self.workspace = Path(workspace_path).expanduser().resolve()

    def _normalize_file_paths(self, file_paths: Union[str, List[str]]) -> List[str]:
        if not file_paths:
            return []
        if isinstance(file_paths, list):
            cleaned = []
            for p in file_paths:
                if isinstance(p, str):
                    cleaned.extend(self._normalize_file_paths(p))
            return cleaned
        if isinstance(file_paths, str):
            raw_str = file_paths.strip()
            if raw_str.startswith('[') and raw_str.endswith(']'):
                try:
                    parsed = json.loads(raw_str)
                    if isinstance(parsed, list):
                        return [str(p).strip() for p in parsed if p]
                except json.JSONDecodeError:
                    pass
                inner = raw_str[1:-1].strip()
                if inner:
                    parts = []
                    for part in inner.split(','):
                        part_clean = part.strip().strip('"').strip("'").strip()
                        if part_clean:
                            parts.append(part_clean)
                    return parts
            if ',' in raw_str and not raw_str.startswith('{'):
                parts = []
                for part in raw_str.split(','):
                    part_clean = part.strip().strip('"').strip("'").strip()
                    if part_clean:
                        parts.append(part_clean)
                return parts
            cleaned_path = raw_str.strip('"').strip("'").strip()
            if cleaned_path:
                return [cleaned_path]
        return []

    def read_files(self, file_paths: Union[str, List[str]]) -> str:
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

    def apply_search_replace_patch(self, file_path: str, patch_block: str) -> str:
        try:
            safe_path = sanitize_and_resolve_path(str(self.workspace), file_path, create_parent=False)
            if not safe_path.exists():
                return f"Lỗi: Không tìm thấy tệp tin '{file_path}' cần áp dụng bản vá."
                
            # SỬA ĐỔI: Tối ưu hóa Regex để xử lý linh hoạt khoảng trắng và ký tự xuống dòng (\r\n hoặc \n)
            pattern = r"<<<<<<<\s*SEARCH\s*[\r\n]+(.*?)(?:[\r\n]+)=======\s*[\r\n]+(.*?)(?:[\r\n]+)>>>>>>>\s*REPLACE"
            match = re.search(pattern, patch_block, re.DOTALL)
            
            if not match:
                return (
                    "Lỗi: Bản vá không đúng định dạng. Bạn bắt buộc phải tuân thủ cấu trúc:\n"
                    "<<<<<<< SEARCH\n"
                    "[Mã nguồn cũ cần tìm chính xác]\n"
                    "=======\n"
                    "[Mã nguồn mới thay thế]\n"
                    ">>>>>>> REPLACE"
                )
                
            search_code = match.group(1).strip("\r\n")
            replace_code = match.group(2).strip("\r\n")
            
            original_content = safe_path.read_text(encoding="utf-8")
            
            # Khớp lỏng hơn bằng cách chuẩn hóa kết thúc dòng khi tìm kiếm
            # (Giúp công cụ bền bỉ hơn khi làm việc với môi trường đa nền tảng)
            normalized_search = search_code.replace("\r\n", "\n")
            normalized_original = original_content.replace("\r\n", "\n")
            
            if normalized_search not in normalized_original:
                return (
                    f"Lỗi: Không tìm thấy phân đoạn mã SEARCH được chỉ định trong tệp '{file_path}'.\n"
                    "Hãy chắc chắn rằng bạn đã copy chính xác từng khoảng trắng và ký tự từ nội dung gốc."
                )
                
            # Thực hiện thay thế và lưu lại cấu trúc xuống dòng ban đầu của tệp
            new_content_normalized = normalized_original.replace(normalized_search, replace_code, 1)
            
            # Trả lại định dạng dòng ban đầu của hệ điều hành
            if "\r\n" in original_content and "\r\n" not in new_content_normalized:
                new_content = new_content_normalized.replace("\n", "\r\n")
            else:
                new_content = new_content_normalized
                
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