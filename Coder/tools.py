# tools.py
import subprocess
from pathlib import Path
from typing import List, Union
from pydantic import BaseModel, Field
from config import GitIgnoreMatcher, sanitize_and_resolve_path

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
            self._run_cmd(["git", "init"])
            self._run_cmd(["git", "config", "user.name", "AI-Agent"])
            self._run_cmd(["git", "config", "user.email", "ai-agent@production.local"])
            try:
                self._run_cmd(["git", "add", "."])
                self._run_cmd(["git", "commit", "-m", "Initial commit from Agent Workspace Setup"])
            except Exception:
                pass
            
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

    def read_files(self, file_paths: Union[str, List[str]]) -> str:
        paths = [file_paths] if isinstance(file_paths, str) else file_paths
        if not paths:
            return "Lỗi: Danh sách đường dẫn tệp tin trống."
            
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

# ==========================================
# SCHEMAS PHỤC VỤ TOOL CALLING
# ==========================================
class ReadFilesSchema(BaseModel):
    file_paths: Union[str, List[str]] = Field(description="Đường dẫn tương đối hoặc danh sách đường dẫn tương đối của các tệp tin.")

class WriteFileSchema(BaseModel):
    file_path: str = Field(description="Đường dẫn tương đối của tệp tin trong workspace cần ghi hoặc cập nhật.")
    content: str = Field(description="Toàn bộ nội dung tệp tin chi tiết cần lưu xuống đĩa.")

class ListDirSchema(BaseModel):
    sub_dir: str = Field(default=".", description="Đường dẫn tương đối của thư mục cần xem.")

class RunTerminalSchema(BaseModel):
    command: str = Field(description="Lệnh terminal hệ điều hành cần thực thi trực tiếp tại thư mục gốc của workspace (ví dụ: 'flutter pub get', 'pytest', 'python main.py').")