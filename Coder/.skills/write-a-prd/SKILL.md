---
name: write-a-prd
description: Thiết lập và viết một Bản Đặc tả Yêu cầu Sản phẩm (PRD.md) hoàn chỉnh tại thư mục gốc của dự án. Quy trình này chuyển hóa ý tưởng thô thành một nguồn tri thức gốc (source of truth) có cấu trúc, mục tiêu đo lường rõ ràng, ranh giới chặt chẽ và chiến lược kiểm thử cụ thể.
---

# Write a PRD

Hãy thực hiện thiết lập một Bản Đặc tả Yêu cầu Sản phẩm (PRD) hoàn chỉnh như một nguồn tri thức gốc (source of truth) cho cả lập trình viên và các vòng kiểm thử tự động, giúp chúng ta có thể bảo lưu bối cảnh một cách an toàn mà không bị phụ thuộc vào lịch sử chat thô.

## Công cụ hỗ trợ trong dự án:
- Sử dụng `write_file` hoặc `apply_search_replace_patch` để tạo/cập nhật tệp `PRD.md` tại thư mục gốc của workspace.
- Sử dụng `read_files` và `search_keyword` để khảo sát mã nguồn hiện hành nhằm đối chiếu tính khả thi của đặc tả.
- Sử dụng `ask_questions_if_underspecified` để thực hiện chất vấn người dùng khi phát hiện điểm thiếu logic hoặc thiếu thông tin.

## Quy trình Thực thi 3 Pha bắt buộc:

### Pha 1: Khảo sát và Chất vấn (Discovery Interview)
Trước khi viết bất kỳ phần nào của PRD, bạn BẮT BUỘC phải đặt câu hỏi hoặc đối thoại (Grilling) để làm rõ các khoảng trống thông tin. Tuyệt đối không tự suy diễn các yếu tố cốt lõi:
- **Nỗi đau thực tế:** Tại sao chúng ta cần xây dựng tính năng này ngay bây giờ?
- **Chỉ tiêu thành công (Success Criteria):** Tránh sử dụng các từ mơ hồ như "nhanh", "mượt mà", "dễ dùng". Phải quy đổi thành các chỉ số đo lường được (ví dụ: thời gian phản hồi API dưới 200ms, tỷ lệ lỗi dưới 0.5%, giao diện đạt điểm Lighthouse tối thiểu 90).

### Pha 2: Phân tích và Định vị Kiến trúc (Scoping)
Đối chiếu yêu cầu mới với cấu trúc thư mục và mã nguồn hiện tại của dự án:
- Tính năng này sẽ tác động đến những module, file, hoặc database schema nào?
- Đâu là các rủi ro hoặc sự phụ thuộc kỹ thuật (technical dependencies) cần được giải quyết?

### Pha 3: Soạn thảo kỹ thuật (Drafting)
Viết tài liệu `PRD.md` vật lý xuống thư mục gốc của dự án theo cấu trúc chuẩn bên dưới.

---

## Nguyên tắc Thiết kế và Ràng buộc khi Soạn thảo:
- **Tuyệt đối không chèn mã nguồn cụ thể** hay đường dẫn tệp tin chi tiết vào PRD để tránh việc tài liệu bị lỗi thời nhanh chóng khi cấu trúc thư mục thay đổi.
- **Ưu tiên Deep Modules:** Thiết kế các phân hệ có chức năng đóng gói mạnh mẽ, che giấu sự phức tạp đằng sau một giao diện (interface) đơn giản, dễ kiểm thử. Tránh thiết kế các module nông (shallow modules) chỉ làm nhiệm vụ trung chuyển dữ liệu đơn giản.
- **Xác định ranh giới rõ ràng:** Phân định rõ những gì nằm ngoài phạm vi phát triển để tránh phình to phạm vi (scope creep).

---

## Cấu trúc chuẩn của PRD.md:

```markdown
# PRD: {Tên tính năng cần phát triển}

## 1. Mô tả bài toán và Chỉ tiêu thành công (Problem Statement & Success Criteria)
- Mô tả ngắn gọn vấn đề của người dùng dưới góc nhìn của họ.
- Các chỉ số đo lường thành công định lượng được (ví dụ: hiệu năng, tỷ lệ lỗi, điểm UI tối thiểu).

## 2. Giải pháp đề xuất (Proposed Solution)
- Giải pháp tổng quan dưới góc nhìn của người dùng (User-facing solution).

## 3. Các câu chuyện người dùng (User Stories)
Hãy viết một danh sách đánh số chi tiết theo định dạng chuẩn:
1. **As a** [vai trò], **I want** [hành động], **so that** [giá trị nhận lại].
2. **As a** ..., **I want** ..., **so that** ...

## 4. Quyết định triển khai và Kiến trúc (Implementation & Architecture Decisions)
- Mô tả các mô-đun chính cần xây dựng hoặc sửa đổi (áp dụng nguyên lý Deep Modules).
- Cấu trúc API chính hoặc mô hình dữ liệu thay đổi (nếu có).
- Rủi ro kỹ thuật và sự phụ thuộc (dependencies) cần lưu ý.

## 5. Quyết định kiểm thử (Testing Decisions)
- Xác định những gì làm nên một bài test tốt cho tính năng này.
- Liệt kê các kịch bản kiểm thử tĩnh hoặc động (bằng CDP/Web test/Unit test) cần phải vượt qua.

## 6. Phạm vi ngoài dự kiến (Out of Scope)
- Chỉ rõ những hành vi, tính năng hoặc trường hợp biên **BẮT BUỘC KHÔNG** thực hiện trong nhiệm vụ lần này để tránh phình to phạm vi phát triển.