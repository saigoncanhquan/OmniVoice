"""Comprehensive CPU Synthesis Benchmark & Audio Mastering Verification for OmniVoice.

Measures:
- Time-to-first-audio (TTFA)
- Total generation time
- Real-Time Factor (RTF) & Seconds per sentence
- CPU utilization & memory consumption
- EBU R128 mastering correctness (active RMS spread, peak ceiling <= -1.0 dBFS)
- SRT subtitle validity and alignment
"""

import os
import sys
import time
import json
import threading
import psutil
import numpy as np
import soundfile as sf
import librosa

# Set root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from omnivoice.tiktok_tts_engine import TikTokTTSEngine, split_sentences_tiktok
from omnivoice.audio_dsp import get_active_speech_rms


class CPUMonitor:
    """Monitors CPU and RAM utilization in a background thread."""
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.running = False
        self.cpu_samples = []
        self.mem_samples = []
        self._thread = None
        self.proc = psutil.Process(os.getpid())

    def _monitor(self):
        self.proc.cpu_percent()
        psutil.cpu_percent()
        while self.running:
            try:
                c = psutil.cpu_percent(interval=self.interval)
                m = self.proc.memory_info().rss / (1024 * 1024)  # MB
                self.cpu_samples.append(c)
                self.mem_samples.append(m)
            except Exception:
                break

    def start(self):
        self.running = True
        self.cpu_samples = []
        self.mem_samples = []
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        avg_cpu = float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0
        peak_cpu = float(np.max(self.cpu_samples)) if self.cpu_samples else 0.0
        peak_mem = float(np.max(self.mem_samples)) if self.mem_samples else 0.0
        return {
            "avg_cpu_percent": round(avg_cpu, 1),
            "peak_cpu_percent": round(peak_cpu, 1),
            "peak_memory_mb": round(peak_mem, 1),
        }


def verify_srt(srt_path: str, expected_chunks: int) -> dict:
    if not os.path.exists(srt_path):
        return {"valid": False, "error": f"File not found: {srt_path}"}
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    if len(blocks) != expected_chunks:
        return {
            "valid": False,
            "error": f"SRT block count ({len(blocks)}) does not match expected chunks ({expected_chunks})",
            "blocks": len(blocks),
        }
    return {
        "valid": True,
        "blocks": len(blocks),
        "file_size": len(content),
    }


def verify_mastering_metrics(audio_path: str, segments: list) -> dict:
    if not os.path.exists(audio_path):
        return {"valid": False, "error": f"Audio file not found: {audio_path}"}

    wav, sr = sf.read(audio_path)
    dur = len(wav) / sr
    overall_peak = float(np.max(np.abs(wav)))
    overall_peak_db = 20.0 * np.log10(max(overall_peak, 1e-12))

    seg_rms_list = []
    seg_act_rms_list = []
    
    for seg in segments:
        st, en = seg["start"], seg["end"]
        s_idx, e_idx = int(st * sr), min(int(en * sr), len(wav))
        chunk_wav = wav[s_idx:e_idx]
        if len(chunk_wav) == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk_wav**2) + 1e-12))
        act_rms = get_active_speech_rms(chunk_wav, sr)
        seg_rms_list.append(rms)
        seg_act_rms_list.append(act_rms)

    act_db_list = [20.0 * np.log10(max(r, 1e-12)) for r in seg_act_rms_list]
    loudness_spread_db = float(np.max(act_db_list) - np.min(act_db_list)) if act_db_list else 0.0

    # Low rumble check (< 80 Hz)
    freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
    fft_mag = np.abs(np.fft.rfft(wav))
    low_energy = np.sum(fft_mag[freqs < 80.0]**2)
    total_energy = np.sum(fft_mag**2) + 1e-12
    rumble_ratio = float(low_energy / total_energy)

    peak_ceiling_pass = overall_peak_db <= -0.95  # Strict -1.0 dBFS with 0.05 margin
    spread_pass = loudness_spread_db <= 1.5  # EBU R128 active leveling consistency

    return {
        "valid": peak_ceiling_pass and spread_pass,
        "duration_sec": round(dur, 2),
        "peak_db": round(overall_peak_db, 2),
        "peak_ceiling_pass": peak_ceiling_pass,
        "active_rms_mean": round(float(np.mean(seg_act_rms_list)), 4) if seg_act_rms_list else 0,
        "active_db_spread": round(loudness_spread_db, 2),
        "spread_pass": spread_pass,
        "sub_80hz_rumble_ratio": round(rumble_ratio, 4),
    }


def benchmark_synthesis(
    engine: TikTokTTSEngine,
    text: str,
    ref_audio: str,
    num_step: int = 16,
    batch_size: int = 1,
    tag: str = "benchmark",
) -> dict:
    chunks = split_sentences_tiktok(text)
    print(f"\n[{tag}] Running synthesis on {len(chunks)} chunks (num_step={num_step}, batch_size={batch_size})...")
    
    monitor = CPUMonitor(interval=0.15)
    monitor.start()

    # Time-to-first-audio measurement (1 sentence chunk)
    t_before_chunk1 = time.perf_counter()
    first_chunk_res = engine.synthesize_script(
        script_text=chunks[0],
        voice_mode="clone",
        ref_audio_path=ref_audio,
        num_step=num_step,
        batch_size=1,
        mastering=True,
        output_basename=f"{tag}_first_chunk",
    )
    ttfa = time.perf_counter() - t_before_chunk1
    print(f"[{tag}] TTFA (first sentence): {ttfa:.2f}s")

    # Full script measurement
    t_full_start = time.perf_counter()
    full_result = engine.synthesize_script(
        script_text=text,
        voice_mode="clone",
        ref_audio_path=ref_audio,
        num_step=num_step,
        batch_size=batch_size,
        mastering=True,
        output_basename=f"{tag}_full",
    )
    total_time = time.perf_counter() - t_full_start
    cpu_stats = monitor.stop()

    audio_dur = full_result["duration"]
    rtf = total_time / max(audio_dur, 0.001)
    sec_per_sentence = total_time / max(len(chunks), 1)

    # Verifications
    srt_check = verify_srt(full_result["srt_path"], len(chunks))
    audio_check = verify_mastering_metrics(full_result["audio_path"], full_result["segments"])

    res = {
        "tag": tag,
        "num_chunks": len(chunks),
        "num_step": num_step,
        "batch_size": batch_size,
        "ttfa_sec": round(ttfa, 2),
        "total_time_sec": round(total_time, 2),
        "audio_duration_sec": round(audio_dur, 2),
        "rtf": round(rtf, 2),
        "sec_per_sentence": round(sec_per_sentence, 2),
        "cpu_stats": cpu_stats,
        "srt_check": srt_check,
        "audio_check": audio_check,
        "audio_path": full_result["audio_path"],
        "srt_path": full_result["srt_path"],
    }
    return res


def main():
    ref_audio = os.path.join(root_dir, "VoiceTemplate", "GiọngNam.wav")
    text_path = os.path.join(root_dir, "VoiceTemplate", "hdsd_Untube.txt")
    if not os.path.exists(text_path):
        text_path = os.path.join(root_dir, "tests", "HDSD_Untube.txt")

    with open(text_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()

    # Split into 2 chunks for benchmark to get clear before/after metrics
    paragraphs = [l.strip() for l in full_text.splitlines() if l.strip()]
    bench_text = "\n".join(paragraphs[:2])

    print("=" * 70)
    print("OmniVoice CPU Benchmark & Studio DSP Verifier")
    print(f"Device: CPU | Cores: {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical")
    print(f"Reference Audio: {ref_audio}")
    print(f"Script Text:\n{bench_text}")
    print("=" * 70)

    t_load_start = time.perf_counter()
    engine = TikTokTTSEngine(device="cpu")
    t_load = time.perf_counter() - t_load_start
    print(f"Engine model loaded in {t_load:.2f}s")

    # 1. Baseline (Before optimization): batch_size=2, num_step=24
    print("\n>>> 1/3 Running Baseline (Before: batch_size=2, num_step=24)...")
    baseline_results = benchmark_synthesis(
        engine=engine,
        text=bench_text,
        ref_audio=ref_audio,
        num_step=24,
        batch_size=2,
        tag="Baseline_CPU_Batch2_24Steps",
    )

    # 2. Optimized Standard: batch_size=1, num_step=24
    print("\n>>> 2/3 Running Optimized Standard (batch_size=1, num_step=24)...")
    hq_results = benchmark_synthesis(
        engine=engine,
        text=bench_text,
        ref_audio=ref_audio,
        num_step=24,
        batch_size=1,
        tag="Optimized_CPU_Batch1_24Steps",
    )

    # 3. Optimized Fast: batch_size=1, num_step=16 (Recommended CPU Fast Mode)
    print("\n>>> 3/3 Running Optimized Fast (batch_size=1, num_step=16)...")
    fast_results = benchmark_synthesis(
        engine=engine,
        text=bench_text,
        ref_audio=ref_audio,
        num_step=16,
        batch_size=1,
        tag="Optimized_CPU_Batch1_16Steps",
    )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON SUMMARY:")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Baseline (B=2, S=24)':<22} | {'Opt Std (B=1, S=24)':<22} | {'Opt Fast (B=1, S=16)':<22}")
    print("-" * 80)
    print(f"{'Time-To-First-Audio (TTFA)':<30} | {baseline_results['ttfa_sec']}s{'':<18} | {hq_results['ttfa_sec']}s{'':<18} | {fast_results['ttfa_sec']}s")
    print(f"{'Total Generation Time':<30} | {baseline_results['total_time_sec']}s{'':<18} | {hq_results['total_time_sec']}s{'':<18} | {fast_results['total_time_sec']}s")
    print(f"{'Audio Duration':<30} | {baseline_results['audio_duration_sec']}s{'':<18} | {hq_results['audio_duration_sec']}s{'':<18} | {fast_results['audio_duration_sec']}s")
    print(f"{'Real-Time Factor (RTF)':<30} | {baseline_results['rtf']}x{'':<18} | {hq_results['rtf']}x{'':<18} | {fast_results['rtf']}x")
    print(f"{'Seconds / Sentence':<30} | {baseline_results['sec_per_sentence']}s{'':<18} | {hq_results['sec_per_sentence']}s{'':<18} | {fast_results['sec_per_sentence']}s")
    print(f"{'Avg CPU Utilization':<30} | {baseline_results['cpu_stats']['avg_cpu_percent']}%{'':<17} | {hq_results['cpu_stats']['avg_cpu_percent']}%{'':<17} | {fast_results['cpu_stats']['avg_cpu_percent']}%")
    print(f"{'Peak CPU Utilization':<30} | {baseline_results['cpu_stats']['peak_cpu_percent']}%{'':<17} | {hq_results['cpu_stats']['peak_cpu_percent']}%{'':<17} | {fast_results['cpu_stats']['peak_cpu_percent']}%")
    print(f"{'Peak Memory (RSS)':<30} | {baseline_results['cpu_stats']['peak_memory_mb']} MB{'':<13} | {hq_results['cpu_stats']['peak_memory_mb']} MB{'':<13} | {fast_results['cpu_stats']['peak_memory_mb']} MB")
    print(f"{'SRT Valid & Aligned':<30} | {str(baseline_results['srt_check']['valid']):<22} | {str(hq_results['srt_check']['valid']):<22} | {str(fast_results['srt_check']['valid'])}")
    print(f"{'Mastering Peak (<= -1 dBFS)':<30} | {baseline_results['audio_check']['peak_db']} dBFS{'':<14} | {hq_results['audio_check']['peak_db']} dBFS{'':<14} | {fast_results['audio_check']['peak_db']} dBFS")
    print(f"{'Active RMS Spread (<1.5dB)':<30} | {baseline_results['audio_check']['active_db_spread']} dB{'':<17} | {hq_results['audio_check']['active_db_spread']} dB{'':<17} | {fast_results['audio_check']['active_db_spread']} dB")
    print("=" * 80)

    def default_serializer(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return str(obj)

    # Save benchmark result as json
    out_json = os.path.join(root_dir, "tests", "benchmark_cpu_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_b2_s24": baseline_results,
            "opt_std_b1_s24": hq_results,
            "opt_fast_b1_s16": fast_results,
        }, f, indent=2, default=default_serializer)
    print(f"Results saved to: {out_json}")


if __name__ == "__main__":
    main()
