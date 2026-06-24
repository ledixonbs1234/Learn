import os
import subprocess
import sys
import tempfile

def run_test_in_sandbox(impl_code:str,test_code:str)->tuple[bool,str]:
    """Chạy code kiểm thử trong một thư mục tạm thời để đảm bảo an toàn.

    Trả về:
        (bool, str): (Trạng thái pass/fail, Toàn bộ log đầu ra từ pytest)
    """
    with tempfile.TemporaryDirectory() as tmpDir:
        # Đường dẫn tới file chứa code logic ứng dụng
        impl_path = os.path.join(tmpDir,'solution.py')
        test_path = os.path.join(tmpDir,'test_solution.py')
        
        
        # Ghi mã nguồn thực thi của Agent xuống đĩa
        with open(impl_path, "w", encoding="utf-8") as f:
            f.write(impl_code)
            
        # Ghi ma nguon xuong dia
        with open(test_path,'w',encoding='utf-8') as f:
            f.write(test_code)
        
        try:
            #Thuc thi lenh pytest trong thu muc tam
            result = subprocess.run(
                [sys.executable,'-m','pytest','test_solution.py'],
                cwd=tmpDir,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            passed = result.returncode ==0
            output = result.stdout + '\n'+ result.stderr
            return passed,output
        
        except subprocess.TimeoutExpired:
            return (
                False,
                "LỖI: Thời gian thực thi vượt quá giới hạn (Timeout - 5 giây). Vui lòng kiểm tra xem code của bạn có bị vòng lặp vô hạn hay không.",
            )
        except Exception as e:
            return False, f"Lỗi hệ thống khi chạy sandbox: {str(e)}"
        
    