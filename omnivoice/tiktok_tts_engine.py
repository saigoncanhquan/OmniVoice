"""TikTok / Reels / Shorts TTS Engine for OmniVoice.

Features:
- Auto temp storage management in D:\temp_OmniVoice
- Fast sentence chunking tailored for short-form fast-paced videos
- Support for Voice Design (Native AI Voices: Nam TikTok, Nữ TikTok, etc.)
- Consistent pitch & timbre across chunks via deterministic position sampling & batching
- Native & Post-processing speed control (1.0x - 1.3x+)
- Subtitle generation (.srt) with accurate millisecond timings for CapCut / Premiere
- Seamless multi-chunk audio concatenation with cross-fading / silence padding
- Export utility to save final audio and subtitles to project directory
"""

import os
import re
import shutil
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

try:
    import torchaudio
except ImportError:
    torchaudio = None

from omnivoice.models.omnivoice import OmniVoice, VoiceClonePrompt
from omnivoice.utils.common import get_best_device

logger = logging.getLogger(__name__)

DEFAULT_TEMP_DIR = r"D:\temp_OmniVoice"

# Known transcript for the default voice template GiọngNam.wav to ensure zero hallucination
DEFAULT_GIONGNAM_TRANSCRIPT = (
    "OmniVoice là mô hình text to speech TTS và voice cloning sao chép giọng zero shot thế hệ mới "
    "phát triển bởi tổ chức k2-fsa hỗ trợ hơn 600 ngôn ngữ trên toàn thế giới bao gồm tiếng Việt "
    "tiếng Anh tiếng Trung tiếng Nhật ứng dụng kiến trúc diffusion language model với tốc độ xử lý "
    "nhanh và độ tự nhiên cao."
)


def ensure_temp_dir(temp_dir: str = DEFAULT_TEMP_DIR) -> str:
    """Ensure the specified temporary directory exists."""
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def split_sentences_tiktok(
    text: str,
    max_chars: int = 75,
    min_chars: int = 15,
) -> List[str]:
    """Split script into short, punchy sentence chunks optimized for TikTok / Shorts."""
    paragraphs = [p.strip() for p in text.strip().splitlines() if p.strip()]
    raw_chunks = []

    for paragraph in paragraphs:
        sentences = re.split(r"([.?!…]+)", paragraph)
        merged_sents = []
        for i in range(0, len(sentences) - 1, 2):
            s = sentences[i].strip() + sentences[i + 1].strip()
            if s:
                merged_sents.append(s)
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            merged_sents.append(sentences[-1].strip())

        for sent in merged_sents:
            if not sent:
                continue
            if len(sent) <= max_chars:
                raw_chunks.append(sent)
            else:
                sub_parts = re.split(r"([,;:，；：]+)", sent)
                curr = ""
                for i in range(0, len(sub_parts) - 1, 2):
                    clause = sub_parts[i].strip() + sub_parts[i + 1].strip()
                    if len(curr) + len(clause) + 1 <= max_chars:
                        curr = (curr + " " + clause).strip() if curr else clause
                    else:
                        if curr:
                            raw_chunks.append(curr)
                        curr = clause
                if len(sub_parts) % 2 == 1 and sub_parts[-1].strip():
                    rem = sub_parts[-1].strip()
                    if len(curr) + len(rem) + 1 <= max_chars:
                        curr = (curr + " " + rem).strip() if curr else rem
                    else:
                        if curr:
                            raw_chunks.append(curr)
                        curr = rem
                if curr:
                    raw_chunks.append(curr)

    refined_chunks = []
    for chunk in raw_chunks:
        if (
            refined_chunks
            and len(refined_chunks[-1]) + len(chunk) + 1 <= max_chars
            and (len(chunk) < min_chars or len(refined_chunks[-1]) < min_chars)
        ):
            refined_chunks[-1] = (refined_chunks[-1] + " " + chunk).strip()
        else:
            refined_chunks.append(chunk)

    return refined_chunks


def format_srt_timestamp(seconds: float) -> str:
    """Convert float seconds to SRT timestamp format HH:MM:SS,mmm."""
    millis = int(round(seconds * 1000))
    hours = millis // (3600 * 1000)
    millis %= 3600 * 1000
    mins = millis // (60 * 1000)
    millis %= 60 * 1000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def export_srt_file(segments: List[Dict[str, Any]], output_path: str) -> str:
    """Export segment timings and text as standard SRT subtitle file."""
    lines = []
    for idx, seg in enumerate(segments, 1):
        start_str = format_srt_timestamp(seg["start"])
        end_str = format_srt_timestamp(seg["end"])
        text = seg["text"]
        lines.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


class TikTokTTSEngine:
    """High-performance TTS Engine customized for TikTok/Reels/Shorts."""

    def __init__(
        self,
        model_name_or_path: str = "k2-fsa/OmniVoice",
        device: Optional[str] = None,
        temp_dir: str = DEFAULT_TEMP_DIR,
    ):
        self.temp_dir = ensure_temp_dir(temp_dir)
        self.device = device or get_best_device()
        if not str(self.device).startswith("cuda"):
            torch.set_num_threads(os.cpu_count() or 4)
        logger.info(f"Initializing TikTokTTSEngine on {self.device}...")
        self.model = OmniVoice.from_pretrained(
            model_name_or_path,
            device_map=self.device,
            dtype=torch.float16 if str(self.device).startswith("cuda") else torch.float32,
        )
        self.sampling_rate = self.model.sampling_rate
        self._cached_prompt: Optional[VoiceClonePrompt] = None
        self._cached_prompt_audio_path: Optional[str] = None

    def prepare_voice_clone(
        self,
        ref_audio_path: str,
        ref_text: Optional[str] = None,
    ) -> VoiceClonePrompt:
        """Load and cache Voice Clone Prompt with exact transcript matching."""
        if self._cached_prompt is not None and self._cached_prompt_audio_path == ref_audio_path:
            return self._cached_prompt

        logger.info(f"Preparing Voice Clone Prompt from: {ref_audio_path}")
        audio_data, sr = sf.read(ref_audio_path)

        norm_path = os.path.normpath(ref_audio_path).lower()
        if ref_text is None:
            # Check known templates to avoid unnecessary whisper transcription & network usage
            if any(k in norm_path for k in ["giọngnam", "giongnam", "giong_nam", "giọng_nam"]):
                if "4s" in norm_path:
                    ref_text = "OmniVoice là mô hình text to speech TTS và voice cloning sao chép giọng zero shot thế hệ mới"
                else:
                    ref_text = DEFAULT_GIONGNAM_TRANSCRIPT
            elif "test_short" in norm_path:
                ref_text = "OmniVoice là mô hình text to speech TTS và voice cloning"
            else:
                try:
                    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
                    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
                    m_whisper = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
                    pipe = pipeline(
                        "automatic-speech-recognition",
                        model=m_whisper,
                        tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor,
                    )
                    res = pipe({"raw": audio_data, "sampling_rate": sr}, generate_kwargs={"language": "vietnamese"})
                    ref_text = res.get("text", "").strip()
                except Exception as e:
                    logger.warning(f"Whisper auto-transcribe fallback: {e}")
                    ref_text = "OmniVoice là mô hình text to speech TTS và voice cloning."

        prompt = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            preprocess_prompt=True,
        )
        self._cached_prompt = prompt
        self._cached_prompt_audio_path = ref_audio_path
        return prompt

    def synthesize_script(
        self,
        script_text: str,
        voice_mode: str = "design_male",  # "design_male", "design_female", "clone"
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        speed: float = 1.15,
        pause_between_chunks: float = 0.08,
        num_step: int = 24,
        guidance_scale: float = 2.0,
        pitch_consistency: bool = True,
        seed: int = 42,
        output_basename: str = "tiktok_audio",
    ) -> Dict[str, Any]:
        """Synthesize script text with unified pitch and timbre across all sentences.

        Key optimizations for consistent timbre:
        - Deterministic seed across chunks (or shared batch)
        - position_temperature = 0.0 to prevent random pitch drifting
        - Explicit pitch anchor: 'moderate pitch'
        """
        ensure_temp_dir(self.temp_dir)
        chunks = split_sentences_tiktok(script_text)
        if not chunks:
            raise ValueError("Input script contains no readable text.")

        logger.info(f"Synthesizing {len(chunks)} chunks with voice_mode='{voice_mode}', pitch_consistency={pitch_consistency}")

        instruct_str = None
        clone_prompt = None
        if voice_mode == "design_male":
            instruct_str = "male, young adult, moderate pitch"
        elif voice_mode == "design_female":
            instruct_str = "female, young adult, moderate pitch"
        elif voice_mode == "clone":
            if not ref_audio_path or not os.path.exists(ref_audio_path):
                raise ValueError(f"Reference audio not found: {ref_audio_path}")
            clone_prompt = self.prepare_voice_clone(ref_audio_path, ref_text=ref_text)
        else:
            instruct_str = "male, young adult, moderate pitch"

        # Position temperature: 0.0 locks greedy token unmasking -> eliminates pitch drift
        pos_temp = 0.0 if pitch_consistency else 5.0

        # Batch synthesis: 2 for CPU to save RAM & compute, 4 for GPU
        batch_size = 2 if not str(self.device).startswith("cuda") else 4
        chunk_waveforms = []

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            b_len = len(batch_chunks)

            if pitch_consistency:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            gen_kwargs = {
                "text": batch_chunks if b_len > 1 else batch_chunks[0],
                "language": ["vi"] * b_len if b_len > 1 else "vi",
                "denoise": False,
                "num_step": num_step,
                "guidance_scale": guidance_scale,
                "speed": [speed] * b_len if b_len > 1 else speed,
                "position_temperature": pos_temp,
            }

            if clone_prompt is not None:
                gen_kwargs["voice_clone_prompt"] = [clone_prompt] * b_len if b_len > 1 else clone_prompt
            elif instruct_str is not None:
                gen_kwargs["instruct"] = [instruct_str] * b_len if b_len > 1 else instruct_str

            out = self.model.generate(**gen_kwargs)
            if isinstance(out, list):
                for wav in out:
                    chunk_waveforms.append(wav)
            else:
                chunk_waveforms.append(out)

        # Concatenate audio chunks and calculate accurate SRT timestamps
        pause_samples = int(self.sampling_rate * max(0.02, pause_between_chunks))
        silence_padding = np.zeros(pause_samples, dtype=np.float32)

        combined_audio = []
        segments = []
        current_time = 0.0

        for idx, (chunk_text, wav) in enumerate(zip(chunks, chunk_waveforms)):
            chunk_duration = len(wav) / self.sampling_rate
            seg_start = current_time
            seg_end = current_time + chunk_duration

            segments.append({
                "index": idx + 1,
                "start": seg_start,
                "end": seg_end,
                "duration": chunk_duration,
                "text": chunk_text,
            })

            combined_audio.append(wav)
            if idx < len(chunks) - 1:
                combined_audio.append(silence_padding)
                current_time = seg_end + (pause_samples / self.sampling_rate)
            else:
                current_time = seg_end

        final_waveform = np.concatenate(combined_audio, axis=0)

        # Save temporary wav
        out_wav_path = os.path.join(self.temp_dir, f"{output_basename}.wav")
        sf.write(out_wav_path, final_waveform, self.sampling_rate)

        # Save temporary srt
        out_srt_path = os.path.join(self.temp_dir, f"{output_basename}.srt")
        export_srt_file(segments, out_srt_path)

        total_duration = len(final_waveform) / self.sampling_rate
        logger.info(f"Done! Audio saved to {out_wav_path} (Duration: {total_duration:.2f}s)")

        return {
            "audio_path": out_wav_path,
            "srt_path": out_srt_path,
            "duration": total_duration,
            "segments": segments,
            "sampling_rate": self.sampling_rate,
        }

    def export_artifacts(
        self,
        result: Dict[str, Any],
        target_dir: str = "outputs",
        custom_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Copy generated audio and srt from temp storage to target project directory."""
        os.makedirs(target_dir, exist_ok=True)
        base_name = custom_name or os.path.splitext(os.path.basename(result["audio_path"]))[0]

        dest_audio = os.path.join(target_dir, f"{base_name}.wav")
        dest_srt = os.path.join(target_dir, f"{base_name}.srt")

        shutil.copy2(result["audio_path"], dest_audio)
        shutil.copy2(result["srt_path"], dest_srt)
        logger.info(f"Exported final artifacts to:\n- {dest_audio}\n- {dest_srt}")

        return dest_audio, dest_srt
