import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
def apply_patch_to_file(file_path:str,search_block:str,replace_block:str)->str:
    """Tìm khối code `search_block` trong file và thay thế nó bằng `replace_block`.
    Hàm này đảm bảo sửa đổi chính xác mà không làm thay đổi các phần khác của file.
    """
    if not os.path.exists(file_path):
        return f"Lỗi: không tìm thấy file '{file_path}'"

    with open(file_path,'r',encoding='utf-8') as f:
        content = f.read()

        
    if search_block.strip() not in content.strip():
        return f"Lỗi: Không tìm thấy đoạn code cần sửa trong file '{file_path}'. Vui lòng đảm bảo sao chép chính xác từng khoảng trắng và dòng của khối code cũ."
    
    new_content = content.replace(search_block,replace_block,1)

    with open(file_path,'w',encoding='utf-8') as f:
        f.write(new_content)
    
    return 'Thành công: Đã áp dụng thành công bản vá'

