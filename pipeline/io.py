import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 48000


class AudioError(RuntimeError):
    """A problem with the user's file that can be shown to them as-is (no paths, no tracebacks)."""


def _resample(audio, sr, target):
    if sr == target:
        return audio
    from math import gcd
    g = gcd(sr, target)
    return resample_poly(audio, target // g, sr // g, axis=1).astype(np.float32)


def _ffmpeg_decode(path, ffmpeg, target_sr):
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "in.wav")
        r = subprocess.run([ffmpeg, "-y", "-i", path, "-ar", str(target_sr), wav],
                           capture_output=True)
        if r.returncode != 0:
            raise AudioError("This file could not be decoded - it may be corrupt or not an audio file.")
        data, sr = sf.read(wav, dtype="float32", always_2d=True)
    return data.T, sr


def _av_decode(path):
    """Decode M4A / MP4 / AAC / WMA and friends with PyAV (bundles its own decoder, no ffmpeg install needed)."""
    import av
    chunks, sr = [], None
    with av.open(path) as c:
        st = c.streams.audio[0]
        rs = av.AudioResampler(format="fltp", layout="stereo" if st.channels >= 2 else "mono")
        for fr in c.decode(st):
            for o in rs.resample(fr):
                chunks.append(o.to_ndarray())
                sr = o.sample_rate
        for o in rs.resample(None):
            chunks.append(o.to_ndarray())
    if not chunks:
        raise AudioError("This file contains no audio.")
    return np.concatenate(chunks, axis=1).astype(np.float32), sr


def load_audio(path, target_sr=TARGET_SR):
    """Load any audio file -> (float32 array [channels, samples], sample rate)."""
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = data.T
    except Exception:
        try:
            audio, sr = _av_decode(path)
            if audio.shape[1] == 0:
                raise AudioError("This file contains no audio.")
            if audio.shape[0] > 2:
                audio = audio[:2]
            return _resample(audio, sr, target_sr), target_sr
        except AudioError:
            raise
        except Exception:
            pass
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise AudioError(
                "This file could not be read. WAV, FLAC, OGG and MP3 are supported directly; "
                "other formats (M4A, AAC...) need ffmpeg installed (winget install ffmpeg). "
                "If it is one of the supported types, the file may be corrupt.")
        audio, sr = _ffmpeg_decode(path, ffmpeg, target_sr)
    if audio.shape[1] == 0:
        raise AudioError("This file contains no audio.")
    if audio.shape[0] > 2:
        audio = audio[:2]
    return _resample(audio, sr, target_sr), target_sr


def mp3_supported():
    """MP3 export works with libsndfile >= 1.1 (no ffmpeg needed) or with ffmpeg."""
    try:
        return "MP3" in sf.available_formats() or shutil.which("ffmpeg") is not None
    except Exception:
        return shutil.which("ffmpeg") is not None


def save_audio(path, audio, sr):
    """Save as WAV, or MP3 when path ends in .mp3."""
    if path.lower().endswith(".mp3"):
        try:
            if "MP3" in sf.available_formats():
                sf.write(path, audio.T, sr, format="MP3")
                return
        except Exception:
            pass
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise AudioError("MP3 export is not available on this computer (install ffmpeg: winget install ffmpeg). "
                             "Choose WAV instead.")
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "out.wav")
            sf.write(wav, audio.T, sr, subtype="PCM_16")
            r = subprocess.run([ffmpeg, "-y", "-i", wav, "-b:a", "192k", path], capture_output=True)
            if r.returncode != 0:
                raise AudioError("MP3 export failed. Choose WAV instead.")
    else:
        sf.write(path, audio.T, sr, subtype="PCM_16")
