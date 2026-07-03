# Coder/.skills/excel-handler/scripts/inspect_excel.py
import sys
import json
import subprocess
import importlib
from pathlib import Path

# ĐẢM BẢO HOÀN TOÀN ĐẦU RA LÀ UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def ensure_dependencies():
    """Tự động cài đặt các thư viện Excel thiếu vào đúng môi trường ảo hiện hành sử dụng uv pip."""
    required = ["pandas", "openpyxl", "xlrd", "xlwt", "xlutils"]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
            
    if not missing:
        return

    installed = False

    # 1. Tầng 1: Sử dụng uv pip cài đặt siêu tốc và cô lập vào .venv hiện tại
    try:
        res = subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable] + missing + ["--quiet"],
            capture_output=True,
            text=True
        )
        if res.returncode == 0:
            installed = True
    except FileNotFoundError:
        # Lệnh 'uv' không khả dụng toàn cục
        pass
    except Exception:
        pass

    # 2. Tầng 2: Dự phòng sử dụng pip tiêu chuẩn của môi trường ảo
    if not installed:
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + missing + ["--quiet"],
                capture_output=True,
                text=True
            )
            if res.returncode == 0:
                installed = True
        except Exception:
            pass

    # 3. Tầng 3: Tái sinh pip bị hỏng bằng ensurepip nội bộ rồi cài đặt lại
    if not installed:
        try:
            import ensurepip
            ensurepip.bootstrap(upgrade=True, default_pip=True)
            res = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + missing + ["--quiet"],
                capture_output=True,
                text=True
            )
            if res.returncode == 0:
                installed = True
        except Exception:
            pass

    # 4. Tầng 4: Dự phòng cài đặt gián tiếp qua trình thông dịch hệ thống bên ngoài
    if not installed:
        for host_python in ["py", "python", "python3"]:
            try:
                subprocess.run(
                    [host_python, "-m", "pip", "install"] + missing + ["--quiet"],
                    capture_output=True,
                    check=True
                )
                all_imported = True
                for pkg in missing:
                    try:
                        importlib.import_module(pkg)
                    except ImportError:
                        all_imported = False
                if all_imported:
                    return
            except Exception:
                pass

ensure_dependencies()
import pandas as pd

def inspect_excel(file_path: str):
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            print(f"Error: File not found at {file_path}")
            sys.exit(1)
            
        xl = pd.ExcelFile(path)
        sheets_info = {}
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, nrows=5)
            sheets_info[sheet] = {
                "columns": list(df.columns),
                "approximate_rows": len(xl.parse(sheet))
            }
        
        print(json.dumps({"file_name": path.name, "sheets": sheets_info}, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error inspecting Excel file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_excel.py <file_path>")
        sys.exit(1)
    inspect_excel(sys.argv[1])