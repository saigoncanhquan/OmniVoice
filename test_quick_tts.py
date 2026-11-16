"""
Mẫu mã nguồn Python sử dụng OmniVoice để chuyển văn bản thành giọng nói (TTS)
và sao chép giọng (Voice Cloning).
"""
import torch
import soundfile as sf
from omnivoice import OmniVoice

def main():
    print("=" * 60)
    print("  OMNIVOICE PYTHON INFERENCE TEST")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Device dang su dung: {device} ({dtype})")

    print("\nDang nap model tu HuggingFace (k2-fsa/OmniVoice)...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype
    )
    print("Nap model thanh cong!")

    # 1. Text-to-Speech Co Ban (Zero-shot)
    prompt_text = "Xin chào, đây là giọng đọc thử nghiệm từ công nghệ OmniVoice k2-fsa."
    print(f"\nDang tong hop giong doc: '{prompt_text}'...")
    
    # Sinh am thanh (sample rate mac dinh 24000 Hz)
    audio = model.generate(text=prompt_text)

    output_path = "output_test_vn.wav"
    sf.write(output_path, audio, samplerate=24000)
    print(f"-> Da luu file am thanh thanh cong tai: {output_path}")

if __name__ == "__main__":
    main()

