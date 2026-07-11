# openwiki_consolidate.py
import subprocess
import sys

def consolidate_with_openwiki():
    print("🧠 [FluxMem Consolidation] Đang kích hoạt OpenWiki Agent để dọn dẹp bộ não toàn cục...")
    
    # Cấu hình Prompt chỉ thị hành động cho OpenWiki Agent
    conflict_prompt = (
        "Bạn là Chuyên viên Phân tích Xung đột Tri thức (FluxMem Conflict Resolver Agent). "
        "Hãy thực hiện các hành động trực tiếp trên file system tại thư mục ~/.openwiki/wiki:\n"
        "1. Đọc và đối chiếu toàn bộ các tệp quy trình (.md) đang có trong thư mục này.\n"
        "2. Phát hiện xem có quy trình nào bị TRÙNG LẶP hoặc XUNG ĐỘT chỉ dẫn kĩ thuật trực tiếp hay không.\n"
        "3. Nếu phát hiện trùng lặp/xung đột, hãy tự động hợp nhất chúng thành một quy trình tối ưu duy nhất. "
        "Ghi đè/tạo tệp mới và di chuyển toàn bộ các tệp cũ bị trùng lặp vào thư mục ~/.openwiki/wiki/archive/ để lưu trữ dự phòng."
    )
    
    try:
        # Gọi trực tiếp CLI của OpenWiki và truyền Prompt hành động
        # Sử dụng Popen để stream trực tiếp log suy nghĩ của OpenWiki ra terminal
        process = subprocess.Popen(
            ["openwiki", conflict_prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1
        )
        
        # Đọc và hiển thị trực tiếp quá trình suy nghĩ/gọi công cụ của OpenWiki Agent
        for line in process.stdout:
            print(f"🤖 [OpenWiki Agent]: {line}", end="")
            
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ [FluxMem Consolidation] OpenWiki Agent đã hoàn tất việc dọn dẹp và hợp nhất bộ nhớ toàn cục thành công.")
        else:
            stderr_output = process.stderr.read()
            print(f"\n❌ Lỗi khi thực thi OpenWiki CLI (Mã lỗi {process.returncode}):\n{stderr_output}")
            
    except FileNotFoundError:
        print("\n❌ Thất bại: Không tìm thấy lệnh 'openwiki' trong biến môi trường hệ thống.")
        print("Vui lòng đảm bảo bạn đã cài đặt OpenWiki CLI toàn cục bằng lệnh: npm install -g openwiki")
    except Exception as e:
        print(f"\n❌ Gặp lỗi hệ thống khi khởi chạy tiến trình: {str(e)}")

if __name__ == "__main__":
    consolidate_with_openwiki()