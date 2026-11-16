# HƯỚNG DẪN SỬ DỤNG OMNIVOICE (k2-fsa) - TIKTOK & SHORTS STUDIO

> **OmniVoice** là mô hình Text-to-Speech (TTS) và Voice Cloning (sao chép giọng) zero-shot thế hệ mới phát triển bởi tổ chức **k2-fsa**, hỗ trợ hơn **600 ngôn ngữ** trên toàn thế giới (bao gồm Tiếng Việt, Tiếng Anh, Tiếng Trung, Tiếng Nhật...), ứng dụng kiến trúc Diffusion Language Model với tốc độ xử lý nhanh và độ tự nhiên cao.

---

## 1. CẤU HÌNH HỆ THỐNG VÀ BỘ NHỚ TRÊN MÁY NÀY

- **Vị trí dự án:** `D:\Drive\Store3\My Drive\Apps\OmniVoice` (Đang chạy trên Google Drive kết nối 4G).
- **Thư mục lưu trữ tác vụ nặng & Temp:** `D:\temp_OmniVoice\`
  - Toàn bộ file âm thanh tạm thời, cache gradio, cache torch và pycache đều được chuyển hướng về đây để **bảo vệ dung lượng Google Drive và tiết kiệm tối đa dữ liệu mạng 4G**.
- **HuggingFace Cache:** Đã lưu trữ tại `D:\temp_OmniVoice\hf_cache` (và liên kết Junction từ `C:\Users\Admin\.cache\huggingface` để giải phóng hoàn toàn ổ C:).
  - Checkpoint `k2-fsa/OmniVoice` (~3.26 GB) và `openai/whisper-tiny` đã sẵn sàng trên ổ D:. Trình khởi động sẽ tự động nhận diện cache có sẵn và tuyệt đối không tải lại.
  - Khi sang máy khác: Nếu máy khác có ổ D: và thư mục `D:\temp_OmniVoice\hf_cache`, ứng dụng sẽ tự động nhận diện. Nếu không có ổ D:, ứng dụng tự động dùng cache mặc định của Windows (`%USERPROFILE%\.cache\huggingface`).
- **Phần cứng hiện tại:** CPU Intel Core i5-7500T (4 Cores / 4 Threads), 32 GB RAM. Đã tối ưu hóa biến môi trường đa luồng PyTorch (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`).
- **Môi trường Python:** Python 3.13 và `.venv` nội bộ đã được cấu hình liên kết thư viện hệ thống đầy đủ (`torch 2.14.0+cpu`, `torchaudio`, `transformers`, `gradio`, `soundfile`, `librosa`, `num2words`, `ffmpeg`).
- **Kho lưu trữ GitHub:** `https://github.com/saigoncanhquan/OmniVoice.git`

---

## 2. CÁCH KHỞI ĐỘNG NHANH (ONE-CLICK)

### Cách 1: Giao diện chuyên nghiệp cho TikTok / Reels / Shorts (Khuyên dùng)
Bấm đúp chuột vào file:
👉 **`D:\Drive\Store3\My Drive\Apps\OmniVoice\launch_shorts_webui.bat`**

**Tính năng đặc biệt:**
1. **Phân đoạn câu thông minh:** Tự động cắt kịch bản thành từng câu ngắn 15 - 75 ký tự theo nhịp điệu video ngắn.
2. **Khóa đồng nhất cao độ (Pitch Consistency):** Khóa góc giọng và timbre, không bị đổi tông hay đổi giọng giữa các câu.
3. **Tuyển chọn giọng đọc:**
   - *Giọng Nam TikTok (Chuẩn AI sạch)*
   - *Giọng Nữ TikTok (Chuẩn AI sạch)*
   - *Giọng Nam Clone Nhanh 4s (Tối ưu CPU)* - Rất nhanh trên CPU (~15s)
   - *Giọng Nam Clone (từ GiọngNam.wav)* - Mẫu 20s
   - *Upload Giọng Mẫu Riêng (Custom Clone)*
4. **Tự động sinh phụ đề chuẩn SRT:** Phụ đề căn chỉnh khớp từng mili-giây, nạp trực tiếp vào CapCut / Premiere / DaVinci Resolve.
5. **Nút xuất thành phẩm:** Xuất file âm thanh `.wav` và phụ đề `.srt` về thư mục `outputs/` của dự án khi bạn hài lòng.

---

### Cách 2: Giao diện WebUI Gradio Gốc (OmniVoice Demo)
Bấm đúp chuột vào file:
👉 **`D:\Drive\Store3\My Drive\Apps\OmniVoice\launch_webui.bat`**
Hoặc chạy lệnh:
```bash
omnivoice-demo --device cpu --no-asr
```

---

### Cách 3: Chạy thử nghiệm nhanh qua Python script
```bash
python test_quick_tts.py
# hoặc chạy bộ kiểm thử TikTok Studio:
python tests/test_tiktok_tts.py
```

---

## 3. LƯU Ý QUAN TRỌNG KHI SỬ DỤNG TRÊN MẠNG 4G & GOOGLE DRIVE

1. **Không tạo các thư mục nặng trong thư mục dự án:** Mọi thao tác trung gian và sinh file thử nghiệm nên để mặc định chạy vào `D:\temp_OmniVoice`.
2. **Khi muốn lưu trữ mã nguồn:** Chỉ commit các file mã nguồn, file cấu hình và tài liệu về kho git `https://github.com/saigoncanhquan/OmniVoice.git`. File sinh tự động trong `outputs/` đã được chặn qua `.gitignore`.
