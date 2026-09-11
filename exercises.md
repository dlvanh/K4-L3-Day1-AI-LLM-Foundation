# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

Với temp càng thấp thì câu trả lời nhận về sẽ ít sáng tạo hơn là các temp cao hơn. Nhưng khi dùng với temp = 1.5 thì câu trả lời được trả bị trả lời linh tinh, không có cấu trúc gì. Với temp = 0.5 và 1.0 thì sẽ sáng tạo hơn và có thêm nhiều facts về các nước khác không chỉ Việt Nam.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

Với chatbot hỗ trợ khách hàng mình sẽ dùng temp nhỏ hơn để câu trả lời không bị linh tinh, với lại sẽ dùng câu trả lời một cách mạch lạc, đúng với trọng tâm câu trả lời. Còn nếu dùng với temp cao thì câu trả lời có thể trả lời không đúng với vấn đề khách hàng hỏi.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

GPT-4o ước tính đắt hơn GPT-4o mini khoảng 20 lần khi chạy thử cùng 1 prompt qua 2 LLM này. Nếu các người dùng phải dùng với 1 task nặng, yêu cầu 1 LLM khoẻ hơn thì nên sử dụng GPT-4o thay vì GPT-4o mini. Còn nếu chỉ sử dụng với mục đích đơn giản như hỏi thăm, trò chuyện hằng ngày, hoặc với yêu cầu không quá phức tạp thì sẽ ưu tiên GPT-4o mini vì chi phí sẽ thấp hơn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

Khi thay đổi system prompt, hai phản hồi thể hiện sự khác biệt rõ rệt về ngôn từ và cách tiếp cận. Với persona "giáo viên tiểu học", mô hình sử dụng tông giọng thân thiện ("Chào em"), từ ngữ đơn giản và ví dụ ẩn dụ gần gũi như "cuốn sổ tay ghi điểm khi chơi trò chơi với bạn bè". Ngược lại, persona "chuyên gia tài chính" sử dụng thuật ngữ chuyên ngành chuẩn xác (decentralized network, hash, sổ cái phân tán) và trình bày theo cấu trúc phân tích kỹ thuật. System prompt đóng vai trò định hình vai trò (persona), tông giọng (tone of voice) và mức độ chuyên sâu của mô hình sao cho phù hợp nhất với đối tượng người nghe.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

Thực nghiệm trên một đoạn văn tiếng Việt 116 từ: công thức ước lượng thô (`số từ / 0.75`) cho ra 154.7 token, trong khi đếm thực tế bằng `tiktoken` (mã hóa `o200k_base` của gpt-4o) ra 129 token, chênh lệch khoảng 16.6%. Tiếng Việt (và các ngôn ngữ không phải tiếng Anh) thường tốn nhiều token hơn tiếng Anh cùng độ dài vì các thuật toán Tokenizer (như BPE) được huấn luyện chủ yếu trên tập dữ liệu tiếng Anh, nơi các từ phổ biến được gán 1 token duy nhất. Đối với tiếng Việt, các từ có dấu thanh và cấu trúc âm tiết rời rạc thường bị chia nhỏ thành nhiều sub-word token hoặc byte-level token.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

Streaming quan trọng nhất khi người dùng tương tác trực tiếp với AI ví dụ như: chatbot, xử lý tài liệu, tóm tắt văn bản,... Còn non-streaming thì phù hợp với các tác vụ yêu cầu tính toán phức tạp, tốn nhiều thời gian, ví dụ như: phân tích dữ liệu, nghiên cứu, báo cáo,...

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

Exponential backoff có lợi thế lớn là giúp làm giảm mật độ truy cập dồn dập vào server bằng cách tăng gấp đôi thời gian chờ sau mỗi lần thử lại (ví dụ: 0.1s → 0.2s → 0.4s), tạo điều kiện cho server đang quá tải có đủ thời gian phục hồi và xử lý hết hàng chờ. Nếu hàng nghìn client cùng retry với delay cố định giống nhau (ví dụ: luôn chờ 1 giây), hiện tượng "Retry Storm" (hoặc Thundering Herd Problem) sẽ xảy ra: toàn bộ hàng nghìn client bị lỗi sẽ cùng lúc dội lại lượng request khổng lồ vào server sau đúng 1 giây. Việc này tạo ra các đợt xung tải (traffic spikes) liên tục theo chu kỳ, khiến server tiếp tục bị quá tải dai dẳng và không thể tự phục hồi.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

Tôi chọn persona là "Trợ lý gia sư lập trình Python và AI kiên nhẫn, thân thiện" với System prompt là: `"Bạn là một trợ lý gia sư lập trình Python và AI kiên nhẫn, thân thiện. Hãy trả lời ngắn gọn, cô đọng dưới 3 đoạn văn, ưu tiên cung cấp ví dụ code minh họa rõ ràng và luôn phản hồi bằng tiếng Việt chuẩn xác."`

Giải thích từ ngữ quan trọng:
- "trả lời ngắn gọn, cô đọng dưới 3 đoạn văn"**: Giúp giới hạn số lượng output token sinh ra, giảm độ trễ (latency) và tiết kiệm chi phí API, đồng thời giúp người học tập trung vào ý chính mà không bị ngợp thông tin.
- "ưu tiên cung cấp ví dụ code minh họa"**: Giúp người học nhanh chóng hình dung cách áp dụng thực tế thay vì chỉ giải thích lý thuyết thuần túy.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

- Hạn chế lớn nhất: Cắt lịch sử cố định còn 3 lượt gần nhất (6 message). Khi hội thoại kéo dài qua 3 lượt, trợ lý sẽ quên mất ngữ cảnh ban đầu (ví dụ: tên người dùng, mục tiêu bài toán đã thống nhất ở các lượt đầu).
- Đề xuất cải thiện: Triển khai cơ chế Conversation Summary Memory (Tóm tắt lịch sử hội thoại).
- Mô tả cách triển khai: Khi số lượt hội thoại vượt quá 3 lượt, thay vì xóa bỏ các message cũ, ta dùng LLM để tóm tắt các lượt chat cũ thành một đoạn văn ngắn gọn (`summary`). Sau đó, chèn đoạn `summary` này vào ngay sau system prompt (`{"role": "system", "content": f"Bối cảnh trước đó: {summary}"}`). Kỹ thuật này giúp mô hình duy trì được toàn bộ ngữ cảnh quan trọng của cả phiên làm việc mà vẫn giữ số lượng token gửi lên ở mức thấp.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
