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
# CÁC CÔNG CỤ DÒ TÌM HỆ THỐNG THỰC TẾ (SENSING TOOLS)
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
            safe_path = sanitize_and_resolve_path(self.workspace_path, file_path, create_parent=False)
            if not safe_path.exists():
                return f"Lỗi: Không tìm thấy tệp '{file_path}'"
            
            if safe_path.is_dir():
                return f"Lỗi: '{file_path}' là một thư mục, vui lòng truyền đường dẫn tệp tin cụ thể."
            
            content = safe_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            total_lines = len(lines)
            
            start = max(1, start_line)
            end = min(total_lines, end_line)
            
            if start > total_lines or start > end:
                return f"Lỗi: Phạm vi dòng yêu cầu ({start_line} - {end_line}) vượt quá giới hạn của tệp (Tệp hiện có {total_lines} dòng)."
            
            selected_lines = lines[start - 1 : end]
            
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
        "Đi kèm với KHOẢNG DÒNG BẮT ĐẦU VÀ KẾT THÚC chính xác tuyệt đối, Chữ ký khai báo đầy đủ (bao gồm cả tham số và kiểu trả về) "
        "và mô tả chức năng được trích xuất trực tiếp từ các comments/docstrings của chúng. "
        "Hỗ trợ tất cả các ngôn ngữ phổ biến (Python, Dart, TS, JS, Go, Rust, C++, Java, C#, Swift, Kotlin)."
    )
    args_schema: Type[BaseModel] = UniversalSearchSchema
    workspace_path: str

    def _clean_block_comments(self, content: str) -> str:
        """Xóa sạch comment khối /* ... */ đa dòng nhưng bảo toàn tuyệt đối số dòng."""
        def replacer(match):
            newlines_count = match.group(0).count("\n")
            return "\n" * newlines_count
        return re.sub(r"/\*.*?\*/", replacer, content, flags=re.DOTALL)

    def _strip_strings_and_line_comments(self, line: str, ext: str) -> str:
        """Xóa sạch comments và chuỗi ký tự để tránh nhiễu ngoặc nhọn."""
        if ext in [".py", ".yaml", ".yml"]:
            clean_line = re.sub(r"#.*$", "", line)
        else:
            clean_line = re.sub(r"//.*$", "", line)

        clean_line = re.sub(r'"(?:[^"\\]|\\.)*"', '""', clean_line)
        clean_line = re.sub(r"'(?:[^'\\]|\\.)*'", "''", clean_line)
        clean_line = re.sub(r"`(?:[^`\\]|\\.)*`", "``", clean_line)
        return clean_line

    def _find_brace_block_end(self, lines: list, start_idx: int, ext: str) -> int:
        """Thuật toán Brace-Matching nâng cao: Khử chuỗi và comment trước khi đếm ngoặc nhọn."""
        brace_count = 0
        found_first_brace = False
        end_idx = start_idx
        
        first_line_clean = self._strip_strings_and_line_comments(lines[start_idx], ext)
        if "=>" in first_line_clean and "{" not in first_line_clean:
            return start_idx + 1

        for idx in range(start_idx, len(lines)):
            line = lines[idx]
            clean_line = self._strip_strings_and_line_comments(line, ext)
            
            if ";" in clean_line and not found_first_brace:
                return idx + 1
            
            for char in clean_line:
                if char == '{':
                    if not found_first_brace:
                        found_first_brace = True
                    brace_count += 1
                elif char == '}':
                    if found_first_brace:
                        brace_count -= 1
                        if brace_count == 0:
                            return idx + 1
            
            if found_first_brace and brace_count == 0:
                return idx + 1
            end_idx = idx
            
        return end_idx + 1

    def _find_python_block_end(self, lines: list, start_idx: int) -> int:
        """Thuật toán Indentation nâng cao: Nhận diện ranh giới thụt lề của Python."""
        sig_end_idx = start_idx
        for idx in range(start_idx, len(lines)):
            clean_line = lines[idx].strip().split("#", 1)[0].strip()
            if clean_line.endswith(":"):
                sig_end_idx = idx
                break
        
        start_line = lines[start_idx]
        start_indent = len(start_line) - len(start_line.lstrip())
        
        end_idx = sig_end_idx
        for idx in range(sig_end_idx + 1, len(lines)):
            line = lines[idx]
            clean_line = line.strip()
            
            if not clean_line or clean_line.startswith('#') or clean_line.startswith('"""') or clean_line.startswith("'''"):
                continue
                
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= start_indent:
                return end_idx + 1
                
            end_idx = idx
            
        return end_idx + 1

    # ==========================================
    # 🛠️ THÊM MỚI: BỘ TRÍCH XUẤT CHỮ KÝ KHAI BÁO TOÀN DIỆN (SIGNATURE EXTRACTOR)
    # ==========================================
    def _extract_full_signature(self, lines: list, start_idx: int, ext: str) -> str:
        """
        Trích xuất và định dạng chuẩn hóa toàn bộ signature khai báo (kể cả khi viết trên nhiều dòng)
        để hiển thị nguyên vẹn kiểu dữ liệu tham số và kiểu trả về trong bảng Markdown.
        """
        sig_end_idx = start_idx
        
        if ext == ".py":
            # Python signature kết thúc bằng dấu ':'
            for idx in range(start_idx, len(lines)):
                clean_line = lines[idx].strip().split("#", 1)[0].strip()
                if clean_line.endswith(":"):
                    sig_end_idx = idx
                    break
        else:
            # C-Style signature kết thúc khi gặp '{', ';' hoặc '=>'
            for idx in range(start_idx, len(lines)):
                clean_line = self._strip_strings_and_line_comments(lines[idx], ext).strip()
                if "{" in clean_line or ";" in clean_line or "=>" in clean_line:
                    sig_end_idx = idx
                    break
                    
        # Gộp các dòng cấu thành signature lại làm một
        sig_parts = []
        for idx in range(start_idx, sig_end_idx + 1):
            line = lines[idx].strip()
            if ext == ".py":
                line = line.split("#", 1)[0].strip()
            else:
                line = line.split("//", 1)[0].strip()
            if line:
                sig_parts.append(line)
                
        full_sig = " ".join(sig_parts)
        
        # Nén khoảng trắng thừa
        full_sig = re.sub(r"\s+", " ", full_sig)
        
        # Loại bỏ decorators/annotations ở đầu (ví dụ: @override, @tool)
        full_sig = re.sub(r"^@\w+(?:\([^)]*\))?\s+", "", full_sig)
        
        # Làm sạch các ký hiệu mở khối code thừa thãi để tập trung vào định nghĩa kiểu
        if ext == ".py":
            if not full_sig.endswith(":"):
                full_sig += ":"
        else:
            # Đối với Dart arrow function, chỉ lấy phần signature trước dấu '=>'
            if "=>" in full_sig:
                full_sig = full_sig.split("=>")[0].strip()
            # Xóa dấu mở ngoặc nhọn hoặc chấm phẩy ở cuối chữ ký C-Style
            full_sig = full_sig.rstrip("{").rstrip(";").strip()
            
        return full_sig

    def _extract_description(self, lines: list, start_line_1based: int, ext: str) -> str:
        """Trích xuất giải thích/mô tả chức năng từ Docstrings hoặc Comments."""
        description_lines = []
        def_idx = start_line_1based - 1

        if ext == ".py":
            docstring_started = False
            quote_char = None
            for idx in range(def_idx + 1, len(lines)):
                line = lines[idx].strip()
                if not line:
                    continue
                
                if not docstring_started:
                    if line.startswith('"""'):
                        quote_char = '"""'
                        docstring_started = True
                        content = line[3:]
                        if content.endswith('"""') and len(line) >= 6:
                            return content[:-3].strip()
                        if content:
                            description_lines.append(content)
                    elif line.startswith("'''"):
                        quote_char = "'''"
                        docstring_started = True
                        content = line[3:]
                        if content.endswith("'''") and len(line) >= 6:
                            return content[:-3].strip()
                        if content:
                            description_lines.append(content)
                    else:
                        break
                else:
                    if line.endswith(quote_char):
                        content = line[:-3]
                        if content:
                            description_lines.append(content)
                        break
                    else:
                        description_lines.append(line)
            
            if description_lines:
                return " ".join([l.strip() for l in description_lines if l.strip()]).strip()

        up_idx = def_idx - 1
        comment_block = []
        in_block_comment = False
        
        while up_idx >= 0:
            line = lines[up_idx].strip()
            if not line:
                if comment_block:
                    break
                up_idx -= 1
                if def_idx - up_idx > 3:
                    break
                continue
            
            if line.startswith("///") or line.startswith("//"):
                cleaned_comment = line.lstrip("/").strip()
                comment_block.insert(0, cleaned_comment)
                up_idx -= 1
            elif line.endswith("*/"):
                in_block_comment = True
                cleaned = line.rstrip("*/").strip()
                if cleaned.startswith("/*"):
                    cleaned = cleaned.lstrip("/*").strip()
                    comment_block.insert(0, cleaned)
                    break
                if cleaned:
                    comment_block.insert(0, cleaned)
                up_idx -= 1
            elif in_block_comment:
                if line.startswith("/*"):
                    cleaned = line.lstrip("/*").strip()
                    if cleaned:
                        comment_block.insert(0, cleaned)
                    in_block_comment = False
                    break
                else:
                    cleaned = line.lstrip("*").strip()
                    comment_block.insert(0, cleaned)
                up_idx -= 1
            else:
                break
                
        if comment_block:
            return " ".join([c.strip() for c in comment_block if c.strip()]).strip()
            
        def_line = lines[def_idx]
        if "#" in def_line and ext == ".py":
            return def_line.split("#", 1)[1].strip()
        elif "//" in def_line and ext != ".py":
            return def_line.split("//", 1)[1].strip()

        return "Không có mô tả"

    # ==========================================
    # PHƯƠNG THỨC THỰC THI CHÍNH (RUN METHOD)
    # ==========================================
    def _run(self, file_path: str) -> str:
        try:
            safe_path = sanitize_and_resolve_path(self.workspace_path, file_path, create_parent=False)
            if not safe_path.exists():
                return f"Thất bại: Không tìm thấy tệp tin '{file_path}'"
            
            raw_content = safe_path.read_text(encoding="utf-8")
            ext = safe_path.suffix.lower()
            
            clean_content = self._clean_block_comments(raw_content)
            lines = clean_content.splitlines()
            
            patterns = {
                ".py": [
                    r"^\s*(class\s+[a-zA-Z0-9_]+)",
                    r"^\s*(?:async\s+)?(def\s+[a-zA-Z0-9_]+)"
                ],
                ".dart": [
                    r"\b(class|mixin|extension)\s+[a-zA-Z0-9_<>]+",
                    r"^\s*(?:async\s+)?(?:[\w<>]+[\s\n]+)+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*(?:async\s*)?\{?",
                    r"^\s*(?:async\s+)?([a-zA-Z0-9_]+)\s*\([^)]*\)\s*(?:async\s*)?\{\s*$",
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
                    r"^\s*(?:[\w&*<>:]+\s+)+([a-zA-Z0-9_~]+::)?([a-zA-Z0-9_~]+)\s*\([^)]*\)\s*(?:const|override|noexcept)?\s*\{?",
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
                ".java": [
                    r"\b(?:public|protected|private|static|\s)+(?:class|interface|enum)\s+([a-zA-Z0-9_]+)",
                    r"^\s*(?:@\w+\s+)*(?:public|protected|private|static|final|synchronized|\s)+(?:[\w<>]+)\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{?"
                ],
                ".cs": [
                    r"\b(?:public|protected|private|internal|static|\s)+(?:class|interface|struct|enum)\s+([a-zA-Z0-9_]+)",
                    r"^\s*(?:public|protected|private|internal|static|virtual|override|async|partial|\s)+(?:[\w<>]+)\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*\{?"
                ],
                ".swift": [
                    r"\b(?:public|internal|private|fileprivate|open|\s)*(?:class|struct|protocol|enum|extension)\s+([a-zA-Z0-9_]+)",
                    r"^\s*(?:public|internal|private|fileprivate|open|static|class|override|async|\s)*func\s+([a-zA-Z0-9_]+)\s*\("
                ],
                ".kt": [
                    r"\b(?:public|internal|private|protected|sealed|data|\s)*(?:class|interface|object)\s+([a-zA-Z0-9_]+)",
                    r"\b(?:public|internal|private|protected|override|actual|expect|inline|tailrec|\s)*fun\s+(?:\([^)]+\)\s*)?([a-zA-Z0-9_]+)\s*\("
                ]
            }

            selected_patterns = patterns.get(ext, [r"\b(class\s+\w+)", r"\b(function\s+\w+)"])
            raw_symbols = []
            
            for line_no, line in enumerate(lines, 1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                
                if clean_line.startswith("//") or clean_line.startswith("#") or clean_line.startswith("/*") or clean_line.startswith("*"):
                    continue
                
                clean_matching_line = self._strip_strings_and_line_comments(line, ext)
                if not clean_matching_line.strip():
                    continue
                
                for pattern in selected_patterns:
                    if re.search(pattern, clean_matching_line):
                        start_line = line_no
                        
                        if ext == ".py":
                            end_line = self._find_python_block_end(lines, line_no - 1)
                        else:
                            end_line = self._find_brace_block_end(lines, line_no - 1, ext)
                        
                        # ⚙️ NÂNG CẤP CHÍNH: Gọi hàm _extract_full_signature để lấy signature nguyên vẹn
                        symbol_signature = self._extract_full_signature(lines, line_no - 1, ext)
                        description = self._extract_description(lines, line_no, ext)
                        
                        raw_symbols.append((start_line, end_line, symbol_signature, description))
                        break
            
            if not raw_symbols:
                return f"Thông báo: Đã quét tệp '{file_path}' nhưng không phát hiện cấu trúc đặc trưng nào."
            
            table_rows = [
                f"### 📊 DANH SÁCH CÁC HÀM TRONG `{file_path}`",
                "",
                "| STT | Chữ ký khai báo (Signature) | Dòng | Mô tả chức năng |",
                "| :---: | :--- | :---: | :--- |"
            ]
            
            for stt, (start, end, signature, desc) in enumerate(raw_symbols, 1):
                # Hiển thị signature trong khối inline code `...` để giữ nguyên định dạng kiểu dữ liệu
                table_rows.append(f"| {stt} | `{signature}` | {start:02d}-{end:02d} | {desc} |")
                
            return "\n".join(table_rows)

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
            
            normalized_search = search_code.replace("\r\n", "\n")
            normalized_original = original_content.replace("\r\n", "\n")
            
            if normalized_search not in normalized_original:
                return (
                    f"Lỗi: Không tìm thấy phân đoạn mã SEARCH được chỉ định trong tệp '{file_path}'.\n"
                    "Hãy chắc chắn rằng bạn đã copy chính xác từng khoảng trắng và ký tự từ nội dung gốc."
                )
                
            new_content_normalized = normalized_original.replace(normalized_search, replace_code, 1)
            
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