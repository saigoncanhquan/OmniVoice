"""Test script for OmniVoice TikTok TTS Engine.

Runs end-to-end synthesis using:
- Reference audio: VoiceTemplate/GiọngNam.wav
- Sample text: VoiceTemplate/hdsd_Untube.txt
- Temp storage: D:\temp_OmniVoice
- Export check: tests/output
"""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from omnivoice.tiktok_tts_engine import TikTokTTSEngine, DEFAULT_TEMP_DIR

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ref_audio = os.path.join(root_dir, "VoiceTemplate", "GiọngNam.wav")
    script_file = os.path.join(root_dir, "VoiceTemplate", "hdsd_Untube.txt")
    export_dir = os.path.join(root_dir, "VoiceTemplate", "output")

    if not os.path.exists(ref_audio):
        print(f"Error: Reference audio not found at {ref_audio}")
        sys.exit(1)

    if not os.path.exists(script_file):
        print(f"Error: Script file not found at {script_file}")
        sys.exit(1)

    with open(script_file, "r", encoding="utf-8") as f:
        full_text = f.read()

    # For fast testing verification, take the first 2-3 sentences or full
    test_short = os.environ.get("TEST_SHORT", "1") == "1"
    if test_short:
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        test_text = "\n".join(lines[:2])
    else:
        test_text = full_text

    print("=" * 60)
    print("OmniVoice TikTok TTS Engine - Test Run")
    print(f"Temp Directory: {DEFAULT_TEMP_DIR}")
    print(f"Reference Audio: {ref_audio}")
    print(f"Input Text:\n{test_text}")
    print("=" * 60)

    t0 = time.time()
    engine = TikTokTTSEngine()

    print(f"Model loaded in {time.time() - t0:.2f}s. Starting synthesis...")
    t1 = time.time()
    result = engine.synthesize_script(
        script_text=test_text,
        ref_audio_path=ref_audio,
        speed=1.2,
        pause_between_chunks=0.08,
        num_step=20,  # Fast high-quality steps
        output_basename="hdsd_Untube_tiktok",
    )
    t2 = time.time()

    print(f"Synthesis finished in {t2 - t1:.2f}s!")
    print(f"Audio Path: {result['audio_path']}")
    print(f"SRT Path: {result['srt_path']}")
    print(f"Total Audio Duration: {result['duration']:.2f}s")
    print(f"Number of Segments: {len(result['segments'])}")

    print("\n--- SRT Content Preview ---")
    with open(result['srt_path'], "r", encoding="utf-8") as f:
        print(f.read())

    # Test export functionality
    print(f"Testing export to {export_dir}...")
    final_audio, final_srt = engine.export_artifacts(
        result,
        target_dir=export_dir,
        custom_name="Untube_tiktok_final",
    )
    print(f"Exported final files:\n- {final_audio}\n- {final_srt}")
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
