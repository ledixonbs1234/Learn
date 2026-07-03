# Coder/.skills/excel-handler/scripts/duplicate_sheet.py
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

def duplicate_sheet(file_path: str, sheet_name: str, new_sheet_name: str = None):
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            print(f"Error: File not found at {file_path}")
            sys.exit(1)
            
        ext = path.suffix.lower()
        if not new_sheet_name:
            new_sheet_name = f"{sheet_name} (2)"
            
        if ext == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(path)
            if sheet_name not in wb.sheetnames:
                print(f"Error: Sheet '{sheet_name}' not found.")
                sys.exit(1)
            source = wb[sheet_name]
            target = wb.copy_worksheet(source)
            target.title = new_sheet_name
            wb.save(path)
            wb.close()
            print(f"Successfully duplicated sheet '{sheet_name}' to '{new_sheet_name}' on '{path.name}' (.xlsx)")
            
        elif ext == ".xls":
            import xlrd
            from xlutils.copy import copy
            
            rb = xlrd.open_workbook(path, formatting_info=True)
            sheet_names = rb.sheet_names()
            if sheet_name not in sheet_names:
                print(f"Error: Sheet '{sheet_name}' not found.")
                sys.exit(1)
                
            wb = copy(rb)
            rs = rb.sheet_by_name(sheet_name)
            ws_dup = wb.add_sheet(new_sheet_name)
            
            # Thực hiện sao chép dữ liệu từ ô sang ô
            for r in range(rs.nrows):
                for c in range(rs.ncols):
                    val = rs.cell_value(r, c)
                    ws_dup.write(r, c, val)
                    
            wb.save(path)
            print(f"Successfully duplicated sheet '{sheet_name}' to '{new_sheet_name}' on '{path.name}' (.xls)")
        else:
            print(f"Error: Unsupported file format '{ext}'")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error duplicating sheet: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python duplicate_sheet.py <file_path> <sheet_name> [new_sheet_name]")
        sys.exit(1)
        
    f_path = sys.argv[1]
    s_name = sys.argv[2]
    new_s_name = sys.argv[3] if len(sys.argv) > 3 else None
    duplicate_sheet(f_path, s_name, new_s_name)