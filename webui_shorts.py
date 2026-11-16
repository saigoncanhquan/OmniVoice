"""
OmniVoice TikTok & Shorts Studio - WebUI
Giao diện tạo giọng đọc chuyên biệt cho video ngắn TikTok, Reels, YouTube Shorts.
Tích hợp trực tiếp với TikTokTTSEngine của omnivoice.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import gradio as gr

# 1. Cơ chế quản lý Cache thông minh: Tự động dùng cache HuggingFace có sẵn
# Nếu máy đã có model sẵn ở cache mặc định (%USERPROFILE%/.cache/huggingface), tuyệt đối không ghi đè HF_HOME để tránh tải lại tốn 4G
default_hf_hub = os.path.expanduser("~/.cache/huggingface/hub/models--k2-fsa--OmniVoice")
temp_d_hf_hub = r"D:\temp_OmniVoice\hf_cache\hub\models--k2-fsa--OmniVoice"

if os.path.exists(default_hf_hub):
    # Dùng cache có sẵn của hệ thống
    pass
elif os.path.exists(temp_d_hf_hub):
    os.environ["HF_HOME"] = r"D:\temp_OmniVoice\hf_cache"
elif os.path.exists("D:\\"):
    os.environ["HF_HOME"] = r"D:\temp_OmniVoice\hf_cache"
    os.makedirs(r"D:\temp_OmniVoice\hf_cache", exist_ok=True)

# File audio tạm thời vẫn lưu tại D:\temp_OmniVoice (nếu có ổ D) để không làm nặng Google Drive
DEFAULT_TEMP_DIR = r"D:\temp_OmniVoice" if os.path.exists("D:\\") else os.path.join(os.environ.get("TEMP", "."), "temp_OmniVoice")
os.makedirs(DEFAULT_TEMP_DIR, exist_ok=True)
if os.path.exists("D:\\"):
    os.environ.setdefault("GRADIO_TEMP_DIR", r"D:\temp_OmniVoice\gradio_temp")
    os.makedirs(r"D:\temp_OmniVoice\gradio_temp", exist_ok=True)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.makedirs("outputs", exist_ok=True)

# 2. Định vị file mẫu & assets
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_VOICE_PATH = SCRIPT_DIR / "VoiceTemplate" / "GiọngNam.wav"
DEFAULT_VOICE_FAST_PATH = SCRIPT_DIR / "VoiceTemplate" / "giong_nam_ref_4s.wav"
DEFAULT_VOICE_FEMALE_PATH = SCRIPT_DIR / "VoiceTemplate" / "GiọngNữ.wav"
DEFAULT_VOICE_FEMALE_FAST_PATH = SCRIPT_DIR / "VoiceTemplate" / "giong_nu_ref_5s.wav"
SAMPLE_TEXT_PATH = SCRIPT_DIR / "VoiceTemplate" / "hdsd_Untube.txt"

# 3. Import backend engine
try:
    from omnivoice.tiktok_tts_engine import TikTokTTSEngine
    HAS_BACKEND = True
except ImportError as err:
    HAS_BACKEND = False
    print(f"[WARNING] Chưa import được TikTokTTSEngine: {err}")

engine_instance = None

def get_engine():
    global engine_instance
    if engine_instance is None:
        if not HAS_BACKEND:
            raise RuntimeError("Backend engine omnivoice.tiktok_tts_engine chưa sẵn sàng.")
        print("Đang khởi tạo TikTokTTSEngine...")
        engine_instance = TikTokTTSEngine(
            temp_dir=DEFAULT_TEMP_DIR
        )
    return engine_instance

def load_sample_script():
    """Nạp nhanh nội dung kịch bản mẫu từ hdsd_Untube.txt"""
    if SAMPLE_TEXT_PATH.exists():
        with open(SAMPLE_TEXT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Biến phai cát ba đê nặng trịch thành trang web tương tác gửi thẳng qua Da-lô, chỉ trong năm giây!"

def process_tts(
    text,
    voice_choice,
    custom_voice,
    speed,
    pause_duration,
    pitch_consistent,
    mastering_enabled,
    num_steps,
    guidance_scale,
    progress=gr.Progress(track_tqdm=True)
):
    """Xử lý tổng hợp giọng đọc TikTok & sinh phụ đề SRT"""
    if not text or not text.strip():
        raise gr.Error("Vui lòng nhập văn bản hoặc kịch bản cần đọc!")

    progress(0.1, desc="Đang nạp mô hình OmniVoice & khởi tạo engine...")
    engine = get_engine()

    # Xác định voice mode
    ref_audio = None
    if "Giọng Nam TikTok" in voice_choice:
        voice_mode = "design_male"
    elif "Giọng Nữ TikTok" in voice_choice:
        voice_mode = "design_female"
    elif "Giọng Nam Truyền Cảm" in voice_choice:
        voice_mode = "design_storyteller"
    elif "Giọng Nữ Năng Động" in voice_choice:
        voice_mode = "design_tiktok_energetic"
    elif "Giọng Nam Clone Nhanh 4s" in voice_choice:
        voice_mode = "clone"
        ref_audio = str(DEFAULT_VOICE_FAST_PATH)
    elif "Giọng Nam Clone Studio" in voice_choice or "từ GiọngNam.wav" in voice_choice:
        voice_mode = "clone"
        ref_audio = str(DEFAULT_VOICE_PATH)
    elif "Giọng Nữ Clone Nhanh 5s" in voice_choice:
        voice_mode = "clone"
        ref_audio = str(DEFAULT_VOICE_FEMALE_FAST_PATH)
    elif "Giọng Nữ Clone Chuẩn" in voice_choice:
        voice_mode = "clone"
        ref_audio = str(DEFAULT_VOICE_FEMALE_PATH)
    elif "Upload Giọng Mẫu Riêng" in voice_choice:
        voice_mode = "clone"
        if not custom_voice or not os.path.exists(custom_voice):
            raise gr.Error("Vui lòng tải lên file âm thanh mẫu để clone giọng!")
        ref_audio = custom_voice
    else:
        voice_mode = "design_male"

    progress(0.2, desc="Đang tổng hợp giọng nói nhịp điệu TikTok...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_basename = f"tiktok_gen_{timestamp}"

    # Gọi synthesize_script từ TikTokTTSEngine
    result = engine.synthesize_script(
        script_text=text,
        voice_mode=voice_mode,
        ref_audio_path=ref_audio,
        speed=float(speed),
        pause_between_chunks=float(pause_duration),
        num_step=int(num_steps),
        guidance_scale=float(guidance_scale),
        pitch_consistency=bool(pitch_consistent),
        mastering=bool(mastering_enabled),
        output_basename=out_basename
    )

    full_audio_path = result["audio_path"]
    srt_path = result["srt_path"]
    duration = result.get("duration", 0.0)
    segments = result.get("segments", [])

    # Đọc nội dung file SRT để hiển thị preview
    srt_content = ""
    if os.path.exists(srt_path):
        with open(srt_path, "r", encoding="utf-8") as f:
            srt_content = f.read()

    info_msg = f"✨ **Hoàn thành xuất sắc!**\n- Thời lượng tổng: **{duration:.2f}s**\n- Số câu / phân đoạn: **{len(segments)}**\n- File lưu tạm tại: `{full_audio_path}`"

    return full_audio_path, srt_content, srt_path, full_audio_path, info_msg

def export_to_project_outputs(audio_path, srt_path):
    """Copy audio và srt từ temp sang thư mục outputs/ của dự án"""
    if not audio_path or not os.path.exists(audio_path):
        return "⚠️ Chưa có file âm thanh để xuất! Hãy bấm 'Tạo Giọng Đọc TikTok' trước.", None, None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    
    dest_audio = out_dir / f"TikTok_Final_{timestamp}.wav"
    dest_srt = out_dir / f"TikTok_Final_{timestamp}.srt"
    
    shutil.copy2(audio_path, dest_audio)
    if srt_path and os.path.exists(srt_path):
        shutil.copy2(srt_path, dest_srt)
        msg = f"✅ **Đã lưu thành phẩm vào dự án:**\n- 🎵 `{dest_audio.resolve()}`\n- 📜 `{dest_srt.resolve()}`"
        return msg, str(dest_audio), str(dest_srt)
    
    msg = f"✅ **Đã lưu thành phẩm vào dự án:**\n- 🎵 `{dest_audio.resolve()}`"
    return msg, str(dest_audio), None

def build_ui():
    with gr.Blocks(title="OmniVoice TikTok Studio") as demo:
        with gr.Row(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 🎙️ OMNIVOICE TIKTOK & SHORTS STUDIO
                **Bộ công cụ tạo giọng đọc & phụ đề tự động siêu tốc cho TikTok, Reels & YouTube Shorts.**
                *Chế độ lưu tạm thông minh: Tự động dùng cache HuggingFace có sẵn - Tiết kiệm tối đa 4G & bộ nhớ.*
                """
            )

        with gr.Row():
            # CỘT TRÁI: NHẬP LIỆU & CÀI ĐẶT
            with gr.Column(scale=5):
                with gr.Group():
                    gr.Markdown("### 1. 📝 Kịch Bản Video Ngắn")
                    script_input = gr.Textbox(
                        label="Nội dung kịch bản (Kèm dấu câu đầy đủ)",
                        placeholder="Dán kịch bản video TikTok vào đây...",
                        lines=7,
                        value=load_sample_script()
                    )
                    with gr.Row():
                        btn_load_sample = gr.Button("📑 Nạp Kịch Bản Mẫu (hdsd_Untube.txt)", size="sm")
                        btn_clear = gr.Button("🧹 Xóa Kịch Bản", size="sm")

                    btn_load_sample.click(fn=load_sample_script, outputs=script_input)
                    btn_clear.click(fn=lambda: "", outputs=script_input)

                with gr.Group():
                    gr.Markdown("### 2. 🗣️ Chọn Giọng Đọc")
                    voice_choice = gr.Dropdown(
                        label="Giọng đọc tuyển chọn cho TikTok / Shorts",
                        choices=[
                            "Giọng Nam TikTok (Chuẩn AI sạch - Studio Male)",
                            "Giọng Nữ TikTok (Chuẩn AI sạch - Studio Female)",
                            "Giọng Nam Truyền Cảm (Storyteller / Podcast)",
                            "Giọng Nữ Năng Động (Energetic TikTok Shorts)",
                            "Giọng Nam Clone Nhanh 4s (Tối ưu CPU)",
                            "Giọng Nam Clone Studio (GiọngNam.wav)",
                            "Giọng Nữ Clone Chuẩn (GiọngNữ.wav)",
                            "Giọng Nữ Clone Nhanh 5s (giong_nu_ref_5s.wav)",
                            "Upload Giọng Mẫu Riêng (Custom Clone)"
                        ],
                        value="Giọng Nam TikTok (Chuẩn AI sạch - Studio Male)",
                        interactive=True
                    )
                    custom_voice = gr.Audio(
                        label="File ghi âm mẫu của bạn (WAV/MP3)",
                        type="filepath",
                        visible=False
                    )

                    def update_voice_ui(choice):
                        if choice == "Upload Giọng Mẫu Riêng (Custom Clone)":
                            return gr.update(visible=True)
                        return gr.update(visible=False)

                    voice_choice.change(fn=update_voice_ui, inputs=voice_choice, outputs=custom_voice)

                with gr.Group():
                    gr.Markdown("### 3. ⚡ Tinh Chỉnh Nhịp Điệu & Đồng Nhất Cao Độ")
                    with gr.Row():
                        speed_slider = gr.Slider(
                            minimum=0.9,
                            maximum=1.4,
                            value=1.15,
                            step=0.05,
                            label="Tốc độ đọc (Shorts: 1.1x - 1.25x)",
                            info="Tăng nhịp đọc nhanh, dồn dập, giữ chân người xem"
                        )
                        pause_slider = gr.Slider(
                            minimum=0.02,
                            maximum=0.4,
                            value=0.08,
                            step=0.02,
                            label="Khoảng nghỉ giữa các câu (giây)",
                            info="Ngắn giúp kịch bản liền mạch không bị khoảng chết"
                        )
                    
                    pitch_toggle = gr.Checkbox(
                        label="🔒 Khóa đồng nhất cao độ & âm sắc (Pitch Consistency)",
                        value=True,
                        info="Giúp toàn bộ các câu trong video giữ đúng một chất giọng và cao độ đồng đều, không bị đổi giọng giữa các câu."
                    )
                    mastering_toggle = gr.Checkbox(
                        label="🎚️ Studio Vocal Mastering & Voice Leveling (Mastering chuẩn phòng thu)",
                        value=True,
                        info="Khử ù 80Hz, EQ làm sáng giọng, san phẳng âm lượng câu và lấp phòng chống ngắt quãng."
                    )

                    with gr.Accordion("⚙️ Cài Đặt Nâng Cao (Diffusion Steps)", open=False):
                        with gr.Row():
                            num_steps = gr.Slider(
                                minimum=16,
                                maximum=40,
                                value=24,
                                step=2,
                                label="Inference Steps (Mặc định 24)",
                                info="24 bước: tốc độ cực nhanh mà âm thanh vẫn mượt mà"
                            )
                            guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=4.0,
                                value=2.0,
                                step=0.5,
                                label="Guidance Scale (Mặc định 2.0)"
                            )

                btn_generate = gr.Button("🚀 Tạo Giọng Đọc TikTok & Phụ Đề", elem_classes=["primary-btn"])

            # CỘT PHẢI: PREVIEW & XUẤT THÀNH PHẨM
            with gr.Column(scale=5):
                with gr.Group():
                    gr.Markdown("### 4. 🎧 Nghe Thử Âm Thanh (Audio Preview)")
                    audio_player = gr.Audio(
                        label="Audio Thành Phẩm",
                        type="filepath",
                        interactive=False,
                        autoplay=False
                    )
                    gen_info = gr.Markdown("*(Chưa có kết quả tổng hợp)*")

                with gr.Group():
                    gr.Markdown("### 5. 📜 Xem Trước Phụ Đề Chuẩn (.srt cho CapCut)")
                    subtitle_preview = gr.Textbox(
                        label="Nội dung phụ đề Subtitle (.srt)",
                        placeholder="Phụ đề đồng bộ khớp thời lượng từng câu sẽ hiển thị tại đây...",
                        lines=7,
                        interactive=False
                    )

                with gr.Group():
                    gr.Markdown("### 6. 📦 Xuất Thành Phẩm Về Thư Mục Dự Án")
                    gr.Markdown("Lưu an toàn vào thư mục `outputs/` của dự án:")
                    with gr.Row():
                        btn_export = gr.Button("💾 Xuất Thành Phẩm (Copy vào outputs/)", elem_classes=["export-btn"])
                    
                    export_status = gr.Markdown("")
                    with gr.Row():
                        file_out_audio = gr.File(label="Tải Audio (.wav)", interactive=False)
                        file_out_srt = gr.File(label="Tải Subtitle (.srt)", interactive=False)

        # Trạng thái file tạm trung gian
        state_temp_audio = gr.State()
        state_temp_srt = gr.State()

        # Sự kiện tạo TTS
        btn_generate.click(
            fn=process_tts,
            inputs=[
                script_input,
                voice_choice,
                custom_voice,
                speed_slider,
                pause_slider,
                pitch_toggle,
                mastering_toggle,
                num_steps,
                guidance_scale
            ],
            outputs=[
                audio_player,
                subtitle_preview,
                state_temp_srt,
                state_temp_audio,
                gen_info
            ]
        )

        # Sự kiện xuất thành phẩm
        btn_export.click(
            fn=export_to_project_outputs,
            inputs=[state_temp_audio, state_temp_srt],
            outputs=[export_status, file_out_audio, file_out_srt]
        )

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OmniVoice TikTok Studio WebUI")
    parser.add_argument("--server_port", type=int, default=7860, help="Cổng chạy server WebUI")
    parser.add_argument("--share", action="store_true", help="Tạo link public Gradio share")
    args = parser.parse_args()

    custom_css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .header-box {
        text-align: center;
        padding: 20px 10px;
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-box h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .primary-btn {
        background: linear-gradient(90deg, #ec4899, #8b5cf6) !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        padding: 12px !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(236, 72, 153, 0.4) !important;
    }
    .primary-btn:hover {
        opacity: 0.95;
        transform: translateY(-1px);
    }
    .export-btn {
        background: linear-gradient(90deg, #10b981, #059669) !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
    }
    """

    app = build_ui()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=args.server_port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=custom_css,
        inbrowser=True
    )
