---
name: domain-modeling
description: Xây dựng và mài sắc mô hình miền (Domain Model) của dự án. Sử dụng khi người dùng muốn cố định thuật ngữ nghiệp vụ (Ubiquitous Language), ghi nhận quyết định kiến trúc quan trọng (ADR), hoặc khi cần đồng bộ thiết kế hệ thống.
---

# Domain Modeling

Hãy chủ động xây dựng và mài sắc mô hình miền của dự án ngay trong quá trình thiết kế. Đây là một hoạt động thực thi tích cực: thách thức các thuật ngữ mơ hồ, phát hiện các kịch bản biên, và ghi chép lại bảng thuật ngữ (glossary) cũng như các quyết định kiến trúc ngay khi chúng được thống nhất.

## Công cụ hỗ trợ trong dự án:
- Sử dụng `read_files`,`read_file_lines`,`list_directory`, `read_file_lines`,hoặc `search_symbols_universal` để khảo sát cấu trúc hiện hành.
- Sử dụng `write_file` hoặc `apply_search_replace_patch` để tạo/cập nhật `CONTEXT.md` và các tệp tin ADR trong thư mục `docs/adr/` ngay trong workspace.
- Sử dụng `ask_questions_if_underspecified` để chất vấn người dùng khi phát hiện mâu thuẫn thuật ngữ.

## Cấu trúc tệp tin tài liệu miền:
Mặc định dự án sử dụng cấu trúc Đơn miền (Single Context):
```
/
├── CONTEXT.md                    ← Bảng thuật ngữ chuyên ngành (Glossary)
└── docs/
    └── adr/                      ← Các quyết định kiến trúc quan trọng
        ├── 0001-auth-model.md
        └── 0002-database-choice.md
```

Hãy tạo các tệp này một cách lười biếng (lazy creation) — chỉ tạo khi thuật ngữ đầu tiên hoặc ADR đầu tiên xuất hiện.

## Các hoạt động bắt buộc trong phiên làm việc:

1. **Thách thức bảng thuật ngữ (Challenge against glossary):**
   Nếu người dùng sử dụng một thuật ngữ mâu thuẫn với ngôn ngữ đã định nghĩa trong `CONTEXT.md`, hãy gọi công cụ `ask_questions_if_underspecified` để làm rõ ngay lập tức.
   
2. **Làm sắc nét ngôn ngữ mơ hồ (Sharpen fuzzy language):**
   Khi người dùng dùng từ chung chung (như "tài khoản"), hãy truy vấn rõ: Họ muốn nói đến "Khách hàng" (Customer) hay "Người dùng hệ thống" (User).

3. **Thảo luận các kịch bản cụ thể (Concrete Scenarios):**
   Đưa ra các tình huống biên thực tế để kiểm tra ranh giới logic. Ví dụ: "Nếu đơn hàng đã hủy nhưng hệ thống thanh toán bị lỗi thì xử lý thế nào?".

4. **Đối chiếu thực tế với mã nguồn (Cross-reference with code):**
   Nếu người dùng phát biểu logic nghiệp vụ mâu thuẫn với code hiện tại, hãy chỉ ra điểm mâu thuẫn và yêu cầu làm rõ đâu là hành vi đúng.

5. **Cập nhật CONTEXT.md inline:**
   Ngay khi một thuật ngữ được thống nhất, hãy dùng `apply_search_replace_patch` hoặc `write_file` để cập nhật tệp `CONTEXT.md` ngay lập tức. Sử dụng định dạng chuẩn trong `CONTEXT-FORMAT.md`.

6. **Đề xuất viết ADR một cách chắt lọc:**
   Chỉ đề xuất tạo ADR (Architectural Decision Record) khi quyết định đó thỏa mãn cả 3 yếu tố:
   - Khó đảo ngược (Hard to reverse) - chi phí thay đổi sau này rất đắt.
   - Gây ngạc nhiên nếu thiếu ngữ cảnh (Surprising without context) - người đọc tương lai sẽ thắc mắc "tại sao lại làm thế?".
   - Là kết quả của sự đánh đổi thực sự (Real trade-off) - có các giải pháp thay thế nhưng bạn đã chọn một vì lý do cụ thể.
   Sử dụng định dạng chuẩn trong `ADR-FORMAT.md`.
