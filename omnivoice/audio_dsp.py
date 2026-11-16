"""Broadcast & Studio-Grade Vocal Mastering DSP Pipeline for OmniVoice.

Provides:
- High-Pass Filter (80 Hz) to eliminate sub-bass rumble and proximity effect.
- Studio De-Mud & Clarity EQ (-2.2 dB @ 300 Hz, +1.8 dB @ 3.6 kHz).
- Phrase-to-phrase Active Speech Loudness Leveler across sentence chunks.
- Acoustic Room Tone Gap Fill & Smooth Cross-Fading (eliminates digital zero vacuum dropouts).
- Zero-clipping Look-Ahead Peak Limiter (-1.0 dBFS ceiling).

Pure NumPy & SciPy implementation for 100% stability and zero native crash risks.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import scipy.signal as signal

logger = logging.getLogger("omnivoice.dsp")

# ---------------------------------------------------------------------------
# Filter and EQ Primitives
# ---------------------------------------------------------------------------

def highpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 80.0,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter to remove rumble (< 80 Hz)."""
    if len(audio) == 0:
        return audio
    nyquist = 0.5 * sample_rate
    normalized_cutoff = min(max(cutoff_hz / nyquist, 0.001), 0.99)
    sos = signal.butter(order, normalized_cutoff, btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def peaking_eq(
    audio: np.ndarray,
    sample_rate: int,
    center_freq_hz: float,
    gain_db: float,
    q: float = 1.0,
) -> np.ndarray:
    """Second-order biquad peaking EQ filter with zero phase distortion."""
    if len(audio) == 0 or abs(gain_db) < 1e-3:
        return audio
    w0 = 2.0 * np.pi * center_freq_hz / sample_rate
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * max(q, 0.1))
    
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A

    sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Speech Leveling & Active RMS Analysis
# ---------------------------------------------------------------------------

def get_active_speech_rms(
    audio: np.ndarray,
    sample_rate: int,
    floor_db: float = -45.0,
) -> float:
    """Measure RMS of active speech portions (excluding pauses and breath tails)."""
    if len(audio) == 0:
        return 0.0
    hop = max(1, int(sample_rate * 0.02))  # 20ms frames
    n_frames = len(audio) // hop
    if n_frames < 2:
        return float(np.sqrt(np.mean(audio**2) + 1e-12))
    
    frames = audio[:n_frames * hop].reshape(n_frames, hop)
    frame_power = np.mean(frames**2, axis=1)
    
    floor_power = 10.0 ** (floor_db / 10.0)
    peak_power = float(np.max(frame_power)) if len(frame_power) > 0 else 0.0
    relative_gate = max(floor_power, peak_power * 10.0 ** (-30.0 / 10.0))
    
    speech_power = frame_power[frame_power > relative_gate]
    if len(speech_power) == 0:
        return float(np.sqrt(np.mean(audio**2) + 1e-12))
    return float(np.sqrt(np.mean(speech_power) + 1e-12))


def level_speech_chunks(
    chunks: List[np.ndarray],
    sample_rate: int,
    target_rms: float = 0.105,
    max_gain_db: float = 6.0,
) -> List[np.ndarray]:
    """Even out chunk-to-chunk loudness while preserving intra-sentence expressive dynamics."""
    if not chunks:
        return []
    
    # Calculate active RMS for each chunk
    rms_list = [get_active_speech_rms(c, sample_rate) for c in chunks]
    valid_rms = [r for r in rms_list if r > 1e-4]
    
    if not valid_rms:
        return chunks
    
    # Target speech level: use target_rms if specified, or median across takes
    eff_target = target_rms if target_rms > 0 else float(np.median(valid_rms))
    max_linear_boost = 10.0 ** (max_gain_db / 20.0)
    min_linear_cut = 10.0 ** (-max_gain_db / 20.0)

    leveled = []
    for chunk, c_rms in zip(chunks, rms_list):
        if c_rms < 1e-4:
            leveled.append(chunk)
            continue
        gain = eff_target / c_rms
        gain = float(np.clip(gain, min_linear_cut, max_linear_boost))
        leveled.append((chunk * gain).astype(np.float32))
        
    return leveled


# ---------------------------------------------------------------------------
# Room Tone Gap Fill & Transition Smoothing
# ---------------------------------------------------------------------------

def apply_edge_fades(audio: np.ndarray, sample_rate: int, fade_ms: float = 12.0) -> np.ndarray:
    """Apply raised-cosine (Hann) fade-in and fade-out at chunk boundaries."""
    n = len(audio)
    fade_n = min(int(sample_rate * (fade_ms / 1000.0)), n // 4)
    if fade_n <= 0:
        return audio
    
    out = audio.copy()
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_n)))
    out[:fade_n] *= ramp
    out[-fade_n:] *= ramp[::-1]
    return out.astype(np.float32)


def _room_tone_profile(chunks: List[np.ndarray]) -> Optional[Tuple[np.ndarray, float]]:
    """Extract ambient room tone spectrum and level from quietest non-silent frames."""
    frame_len = 1024
    blocks = []
    for c in chunks:
        n = len(c) // frame_len
        if n > 0:
            blocks.append(c[:n * frame_len].reshape(n, frame_len))
    if not blocks:
        return None
    
    frames = np.concatenate(blocks, axis=0)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    
    # Find quiet frames: between -70 dB and -40 dB
    live = (rms > 10.0 ** (-70.0 / 20.0)) & (rms < 10.0 ** (-40.0 / 20.0))
    if np.sum(live) < 4:
        return None
    
    quiet_frames = frames[live & (rms <= np.percentile(rms[live], 25))]
    if len(quiet_frames) < 2:
        return None
    
    level = float(np.sqrt(np.mean(quiet_frames**2)))
    window = np.hanning(frame_len)
    mag = np.mean(np.abs(np.fft.rfft(quiet_frames * window, axis=1)), axis=0)
    return mag, level


def _synthesize_room_tone(profile: Tuple[np.ndarray, float], n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate n_samples of shaped noise matching the room tone profile."""
    mag, level = profile
    rng = np.random.default_rng(seed)
    noise_raw = rng.standard_normal(n_samples)
    spec = np.fft.rfft(noise_raw)
    shape = np.interp(
        np.linspace(0.0, 1.0, spec.shape[0]),
        np.linspace(0.0, 1.0, mag.shape[0]),
        mag,
    )
    shaped = np.fft.irfft(spec * shape, n_samples).astype(np.float32)
    s_rms = float(np.sqrt(np.mean(shaped**2) + 1e-12))
    return (shaped * (level / s_rms)).astype(np.float32)


# ---------------------------------------------------------------------------
# Look-Ahead Peak Limiter
# ---------------------------------------------------------------------------

def _window_min(values: np.ndarray, before: int, after: int) -> np.ndarray:
    """van Herk/Gil-Werman block-wise prefix and suffix minima in O(n)."""
    width = before + after + 1
    n = values.shape[0]
    tail = (-(n + width - 1)) % width
    padded = np.pad(values, (before, after + tail), constant_values=np.inf)
    blocks = padded.reshape(-1, width)
    prefix = np.minimum.accumulate(blocks, axis=1).ravel()
    suffix = np.minimum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].ravel()
    start = np.arange(n)
    return np.minimum(suffix[start], prefix[start + width - 1])


def peak_limit(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -1.0,
    lookahead_ms: float = 4.0,
    release_ms: float = 30.0,
) -> np.ndarray:
    """Look-ahead peak limiter: transparent ceiling at threshold_db."""
    if len(audio) == 0:
        return audio
    
    ceiling = 10.0 ** (threshold_db / 20.0)
    if float(np.max(np.abs(audio))) <= ceiling:
        return audio.astype(np.float32)
    
    flat = audio.astype(np.float64)
    n = len(flat)
    ramp = max(1, int(sample_rate * lookahead_ms / 2000.0))
    hold = max(0, int(sample_rate * release_ms / 1000.0))
    before, after = 2 * ramp + hold, 2 * ramp
    
    gain = np.minimum(1.0, ceiling / np.maximum(np.abs(flat), 1e-12))
    held = _window_min(gain, ramp + hold, ramp)
    csum = np.concatenate(([0.0], np.cumsum(np.pad(held, ramp, mode="edge"))))
    smooth = (csum[2 * ramp + 1:] - csum[:-(2 * ramp + 1)]) / (2 * ramp + 1)
    applied = np.minimum(smooth, gain)
    
    return (flat * applied).astype(np.float32)


# ---------------------------------------------------------------------------
# Master Vocal Studio Pipeline
# ---------------------------------------------------------------------------

def master_vocal_pipeline(
    chunk_waveforms: List[np.ndarray],
    sample_rate: int,
    pause_duration_s: float = 0.08,
    target_rms: float = 0.105,
    highpass_cutoff: float = 80.0,
    demud_freq: float = 300.0,
    demud_gain_db: float = -2.2,
    presence_freq: float = 3600.0,
    presence_gain_db: float = 1.8,
    peak_ceiling_db: float = -1.0,
    enable_room_tone: bool = True,
) -> Tuple[np.ndarray, List[float]]:
    """Master a series of sentence chunk waveforms into a broadcast-ready track.

    Returns:
        (mastered_waveform, chunk_durations)
    """
    if not chunk_waveforms:
        return np.zeros(0, dtype=np.float32), []
    
    # 1. Clean individual chunks: sanitize NaN/inf
    sanitized_chunks = []
    for c in chunk_waveforms:
        c_clean = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        # Apply gentle highpass & de-mud/presence EQ per chunk
        c_hp = highpass_filter(c_clean, sample_rate, cutoff_hz=highpass_cutoff)
        c_eq1 = peaking_eq(c_hp, sample_rate, center_freq_hz=demud_freq, gain_db=demud_gain_db, q=1.2)
        c_eq2 = peaking_eq(c_eq1, sample_rate, center_freq_hz=presence_freq, gain_db=presence_gain_db, q=1.0)
        sanitized_chunks.append(c_eq2)
    
    # 2. Dynamic Speech Leveling across all chunks
    leveled_chunks = level_speech_chunks(
        sanitized_chunks,
        sample_rate=sample_rate,
        target_rms=target_rms,
        max_gain_db=6.0,
    )
    
    # 3. Apply smooth edge fades to prevent clicks
    smoothed_chunks = [apply_edge_fades(c, sample_rate, fade_ms=12.0) for c in leveled_chunks]
    
    # 4. Concatenate with room tone / smooth pause padding
    pause_samples = max(int(sample_rate * 0.02), int(sample_rate * pause_duration_s))
    rt_profile = _room_tone_profile(smoothed_chunks) if enable_room_tone else None
    
    combined = []
    chunk_durations = []
    
    for idx, chunk in enumerate(smoothed_chunks):
        combined.append(chunk)
        chunk_durations.append(len(chunk) / sample_rate)
        
        if idx < len(smoothed_chunks) - 1:
            if rt_profile is not None:
                pause_audio = _synthesize_room_tone(rt_profile, pause_samples, seed=100 + idx)
                # Fade pause audio smoothly at its edges
                p_fade = min(int(sample_rate * 0.015), pause_samples // 3)
                if p_fade > 0:
                    ramp = np.linspace(0.0, 1.0, p_fade)
                    pause_audio[:p_fade] *= ramp
                    pause_audio[-p_fade:] *= ramp[::-1]
            else:
                # Soft comfort dither floor (-60 dBFS) to avoid absolute digital zero vacuum
                comfort_floor = 10.0 ** (-60.0 / 20.0)
                rng = np.random.default_rng(100 + idx)
                pause_audio = (rng.standard_normal(pause_samples) * comfort_floor * 0.5).astype(np.float32)
                p_fade = min(int(sample_rate * 0.015), pause_samples // 3)
                if p_fade > 0:
                    ramp = np.linspace(0.0, 1.0, p_fade)
                    pause_audio[:p_fade] *= ramp
                    pause_audio[-p_fade:] *= ramp[::-1]
            combined.append(pause_audio)
            
    full_waveform = np.concatenate(combined, axis=0)
    
    # 5. Final Look-Ahead Brickwall Peak Limiter
    mastered_waveform = peak_limit(full_waveform, sample_rate, threshold_db=peak_ceiling_db)
    
    return mastered_waveform, chunk_durations
