# Coder/.skills/excel-handler/scripts/read_excel.py
import sys
import subprocess
import importlib
from pathlib import Path

# ĐẢM BẢO HOÀN TOÀN ĐẦU RA LÀ UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def ensure_dependencies():
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

    # 1. Tầng 1: Sử dụng uv pip cài đặt trực tiếp vào .venv hiện tại
    try:
        res = subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable] + missing + ["--quiet"],
            capture_output=True,
            text=True
        )
        if res.returncode == 0:
            installed = True
    except FileNotFoundError:
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

    # 3. Tầng 3: Tái sinh pip bằng ensurepip
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

    # 4. Tầng 4: Dự phòng trình thông dịch bên ngoài
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

def read_excel(file_path: str, sheet_name: str, start_row: int = 0, num_rows: int = 20):
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            print(f"Error: File not found at {file_path}")
            sys.exit(1)
            
        df = pd.read_excel(path, sheet_name=sheet_name)
        total_rows = len(df)
        end_row = min(start_row + num_rows, total_rows)
        
        sliced_df = df.iloc[start_row:end_row]
        markdown_table = sliced_df.to_markdown(index=True)
        
        print(f"=== SHEET: {sheet_name} (Rows {start_row} to {end_row - 1} of {total_rows}) ===\n")
        print(markdown_table)
    except Exception as e:
        print(f"Error reading Excel sheet: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python read_excel.py <file_path> <sheet_name> [start_row] [num_rows]")
        sys.exit(1)
    
    f_path = sys.argv[1]
    s_name = sys.argv[2]
    s_row = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    n_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    read_excel(f_path, s_name, s_row, n_rows)