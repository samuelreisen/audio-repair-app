"""Audio repair pipeline.

    analyse -> declick -> tame -> declip -> (de-reverb + denoise) -> tone EQ -> master

* analyse   : a cheap measurement of the input (clipping, noise floor, hum, reverb, bandwidth,
              loudness) - pipeline/analysis.py.  It decides which stages are worth running and
              how hard, unless the caller forced a value.
* declip    : AR / EM reconstruction of clipped peaks, headroom to -1 dBTP.
* denoise   : mains-hum notches -> WPE de-reverb (scaled by detected reverberance) -> classical +
              RNNoise gain mask, faded out when the noise is already inaudible.
* eq        : adaptive tone correction, bandwidth extension, de-esser, slow leveler.
* master    : gated-LUFS normalisation (-16 voice / -14 music) and true-peak limiting (-1 dBTP).

The order was chosen by ablation (see README, "Stage order").  Options (all optional):

    declick, declip, denoise, eq, master  bool   stage enabled (default True).  An enabled stage is still skipped
                                           automatically when the analysis says it would do nothing.
    denoise_strength                float  0..1.  Absent / None = chosen automatically from the analysis.
    auto                            bool   False = never skip a stage automatically (default True).
    report                          list   if given, human-readable lines about what was detected / done
                                           are appended to it.
    analysis                        dict   if given, it is filled with the analysis measurements.
    order                           tuple  advanced / testing: stage order, any of declip, dereverb,
                                           denoise, level, eq, master.  "dereverb" runs WPE as its own stage
                                           (denoise then does not); "level" runs the speech leveler as its
                                           own stage (eq then does not).
"""
import time

import numpy as np

from .io import load_audio, save_audio
from .declick import declick
from .tame import tame
from .studio import studio_available, studio_enhance
from .declip import declip
from .denoise import denoise
from .dereverb import dereverb
from .voice_eq import voice_eq, voice_eq_with_plan, level_gain, _CENT
from .master import master, master_with_info, measure_lufs
from .analysis import analyse, describe, auto_denoise_strength

DEFAULT_OPTS = {
    "declick": True,
    "tame": True,
    "declip": True,
    "studio": False,               # optional AI voice restoration (slow, voice only); see pipeline/studio.py
    "denoise": True,
    "denoise_strength": None,      # None = automatic
    "eq": True,
    "master": True,
    "auto": True,
}

STAGE_ORDER = ("declick", "tame", "declip", "studio", "denoise", "eq", "master")

_PROGRESS = {"analyse": (0.05, "Analysing the recording"),
             "declick": (0.08, "Removing clicks and pops"),
             "tame": (0.09, "Taming bumps and bangs"),
             "declip": (0.10, "Repairing clipping"),
             "studio": (0.30, "Rebuilding the voice (AI)"),
             "dereverb": (0.30, "Reducing room reverb"),
             "denoise": (0.35, "Removing background noise"),
             "level": (0.65, "Evening out levels"),
             "eq": (0.72, "Fixing tone / mic quality"),
             "master": (0.86, "Mastering")}

__all__ = ["process_array", "process_with_report", "run", "load_audio", "save_audio", "analyse", "describe",
           "DEFAULT_OPTS", "STAGE_ORDER"]


def _floor_dbfs(a):
    """10th percentile of 50 ms frame RMS (dBFS) of the mono mix."""
    m = np.asarray(a, np.float64).mean(axis=0)
    n = 2400
    k = len(m) // n
    if k < 4:
        return None
    r = np.sqrt(np.mean(m[:k * n].reshape(k, n) ** 2, axis=1)) + 1e-9
    return float(np.percentile(20 * np.log10(r), 10))


def _eq_summary(plan):
    """One sentence about what the tone stage did, from voice_eq's plan."""
    if plan.get("skipped"):
        return "tone stage skipped (clip too short or constant)"
    parts = []
    hum = plan.get("hum")
    if hum:
        parts.append("hum notches at %s Hz" % ", ".join("%.0f" % f for f, _, _ in hum[:3]))
    g = plan.get("g")
    if g is not None and len(g) == len(_CENT):
        gg = np.where(_CENT <= 10000.0, g, 0.0)          # the audible speech range only
        i, j = int(np.argmax(gg)), int(np.argmin(gg))
        if gg[i] >= 1.5:
            parts.append("lifted %.1f kHz by %.0f dB" % (_CENT[i] / 1000.0, gg[i]))
        if gg[j] <= -1.5:
            parts.append("cut %.0f Hz by %.0f dB" % (_CENT[j], -gg[j]))
    hpf = plan.get("hpf")
    if hpf and hpf > 40:
        parts.append("rumble filter at %.0f Hz" % hpf)
    bw = plan.get("bwe")
    if bw is not None and bw.get("S") is not None and np.max(bw["S"]) > 0:
        parts.append("restored top end above %.1f kHz" % (bw["f_edge"] / 1000.0))
    if plan.get("deess_max_db", 0) and plan["deess_max_db"] >= 0.5:
        parts.append("de-essed (up to %.1f dB)" % plan["deess_max_db"])
    if plan.get("level_max_db", 0) and plan["level_max_db"] >= 0.5:
        parts.append("levelled (up to %.1f dB)" % plan["level_max_db"])
    return "tone " + ("; ".join(parts) if parts else "already balanced, only hygiene filtering applied")


def process_with_report(audio, sr, mode="voice", opts=None, progress=None):
    """Run the pipeline.  Returns (audio float32 (channels, samples), report lines list, analysis dict)."""
    rep = []
    o = dict(opts or {})
    o["report"] = rep
    an = {}
    o["analysis"] = an
    out = process_array(audio, sr, mode, o, progress)
    return out, rep, an


def process_array(audio, sr, mode="voice", opts=None, progress=None):
    """audio: float32 array shape (channels, samples). Returns processed array.

    See the module docstring for the options.  Output has the same shape and is sample-aligned
    with the input."""
    o = {**DEFAULT_OPTS, **(opts or {})}
    auto = bool(o.get("auto", True))
    rep = o.get("report") if isinstance(o.get("report"), list) else None
    an_out = o.get("analysis") if isinstance(o.get("analysis"), dict) else None
    prog = lambda key: progress and progress(*_PROGRESS[key])

    def say(kind, msg):
        if rep is not None:
            rep.append("%s: %s" % (kind, msg))

    x = np.asarray(audio)
    one_d = x.ndim == 1
    if one_d:
        x = x[None, :]
    x = x.astype(np.float32, copy=False)
    t_all = time.time()

    order = tuple(o.get("order") or STAGE_ORDER)
    for st in order:
        if st not in _PROGRESS or st == "analyse":
            raise ValueError("unknown stage %r (choose from declip, dereverb, denoise, level, eq, master)" % (st,))

    prog("analyse")
    an = analyse(x, sr, mode)
    if an_out is not None:
        an_out.update(an)
    if rep is not None:
        for line in describe(an):
            say("Detected", line)
    if an["silent"]:
        out = x
        return out[0] if one_d else out

    if an["too_short"]:
        say("Skipped", "repair stages (the clip is shorter than half a second)")
        order = tuple(s for s in order if s == "master")
    if o.get("studio") and mode == "voice" and "studio" in order and studio_available():
        # Studio voice rebuilds the whole voice. Measured on the bench, it does clearly worse when clipping repair runs
        # before it (STOI 0.66 -> 0.54) or tone / noise stages run after it (-> 0.35), so it replaces them: only click
        # removal and bump control (harmless in front of it) and the final mastering remain.
        dropped = [s_ for s_ in order if s_ not in ("declick", "tame", "studio", "master")]
        order = tuple(s_ for s_ in order if s_ in ("declick", "tame", "studio", "master"))
        if dropped:
            say("Skipped", "clipping repair, noise reduction and tone correction (Studio voice replaces them)")
    explicit_dereverb = "dereverb" in order
    studio_done = False
    explicit_level = "level" in order
    voice = mode == "voice"

    for st in order:
        # ------------------------------------------------------------------ declick
        if st == "declick":
            if not o["declick"]:
                say("Skipped", "click removal (switched off)")
                continue
            prog("declick")
            x, nclk = declick(x, sr)
            if nclk:
                say("Applied", "click removal: %d clicks / pops repaired" % nclk)
            else:
                say("Skipped", "click removal (no clicks found)")
        # ------------------------------------------------------------------ tame (bangs / thumps)
        elif st == "tame":
            if not o["tame"]:
                say("Skipped", "bump control (switched off)")
                continue
            prog("tame")
            x, nb, cut = tame(x, sr)
            if nb:
                say("Applied", "bump control: %d knocks / bangs turned down (up to %.0f dB)" % (nb, cut))
            else:
                say("Skipped", "bump control (nothing far above the speech level)")
        # ------------------------------------------------------------------ declip
        elif st == "declip":
            if not o["declip"]:
                say("Skipped", "clipping repair (switched off)")
                continue
            if auto and an["clip_kind"] is None:
                say("Skipped", "clipping repair (no clipping detected)")
                continue
            prog("declip")
            x = declip(x)
            say("Applied", "clipping repair on %.1f%% of the samples; peaks rebuilt, headroom restored"
                % an["clip_pct"])
        # ------------------------------------------------------------------ studio voice (optional, AI, voice only)
        elif st == "studio":
            if not o.get("studio") or not voice:
                continue
            if not studio_available():
                say("Skipped", "Studio voice (the AI model is not installed)")
                continue
            prog("studio")

            def _studio_progress(f):
                if progress:
                    progress(0.30 + 0.40 * f, "Rebuilding the voice (AI) %d%%" % int(100 * f))
            x = studio_enhance(x, sr, _studio_progress).astype(np.float32)
            studio_done = True
            say("Applied", "Studio voice: the voice was rebuilt with an AI model (Resemble Enhance) and blended "
                           "with the cleaned original")
        # ------------------------------------------------------------------ dereverb (explicit stage)
        elif st == "dereverb":
            if not o["denoise"] or not voice:
                continue
            if auto and an["reverb"] < 0.05:
                say("Skipped", "de-reverb (room sounds dry)")
                continue
            prog("dereverb")
            x = dereverb(x, sr, strength=0.8, auto=True)
            say("Applied", "de-reverb scaled to the detected room (%.0f%%)" % (100 * an["reverb"]))
        # ------------------------------------------------------------------ denoise
        elif st == "denoise":
            if not o["denoise"]:
                say("Skipped", "noise reduction (switched off)")
                continue
            if studio_done:
                say("Skipped", "noise reduction (the Studio voice model already removed the noise)")
                continue
            forced = o.get("denoise_strength") is not None
            strength = float(o["denoise_strength"]) if forced else auto_denoise_strength(an, mode)
            if strength <= 0.0:
                say("Skipped", "noise reduction (strength 0)")
                continue
            if auto and voice and not forced:
                if an["no_pauses"] and not an["hum_hz"]:
                    # no pauses: the "floor" is soft phonemes, and the reverb detector misreads continuous
                    # speech too, so there is nothing trustworthy to remove (do no harm)
                    say("Skipped", "noise reduction (no steady noise floor: the speaker never pauses)")
                    continue
                audible = (an["presence"] or 0.0) > 0.02
                if not audible and not an["hum_hz"] and (explicit_dereverb or an["reverb"] < 0.05):
                    if an["floor_dbfs"] <= -100.0:
                        say("Skipped", "noise reduction (no noise floor to remove)")
                    else:
                        say("Skipped", "noise reduction (noise already %.0f dB below the speech - inaudible)"
                            % an["dyn_db"])
                    continue
            prog("denoise")
            before = _floor_dbfs(x)
            x = denoise(x, sr, mode, strength, dereverb=not explicit_dereverb)
            after = _floor_dbfs(x)
            bits = ["strength %d%%%s" % (round(100 * strength), "" if forced else " (auto)")]
            if an["hum_hz"]:
                bits.append("%.0f Hz hum removed" % an["hum_hz"])
            if voice and not explicit_dereverb and an["reverb"] >= 0.05:
                bits.append("room reverb reduced")
            if before is not None and after is not None and before > -100:
                bits.append("noise floor %.0f -> %.0f dBFS" % (before, after))
            say("Applied", "noise reduction: " + ", ".join(bits))
        # ------------------------------------------------------------------ pre-leveler (explicit stage)
        elif st == "level":
            if not o["eq"] or not voice or x.shape[1] < int(2 * sr):
                continue
            prog("level")
            gl = level_gain(x.astype(np.float64).mean(0), sr)
            if gl is not None:
                x = (x * 10 ** (gl / 20.0)[None, :]).astype(np.float32)
                say("Applied", "level riding up to %.1f dB" % float(np.abs(gl).max()))
        # ------------------------------------------------------------------ tone
        elif st == "eq":
            if not o["eq"]:
                say("Skipped", "tone correction (switched off)")
                continue
            prog("eq")
            x, plan = voice_eq_with_plan(x, sr, mode, level=not explicit_level,
                                         bwe=(not auto) or not an["no_pauses"])
            say("Applied", _eq_summary(plan))
        # ------------------------------------------------------------------ master
        elif st == "master":
            if not o["master"]:
                say("Skipped", "mastering (switched off)")
                continue
            prog("master")
            x, info = master_with_info(x, sr, mode)
            tgt = -16.0 if mode == "voice" else -14.0
            msg = "mastered to %.1f LUFS with true peak %.1f dBTP" % (info.get("lufs_out", float("nan")),
                                                                    info.get("tp", float("nan")))
            if info.get("depth_limited"):
                msg += " (stopped short of %.0f LUFS: reaching it would have crushed the dynamics)" % tgt
            say("Applied", msg)
        else:
            raise ValueError("unknown stage %r" % (st,))

    if progress:
        progress(0.95, "Saving")
    if an_out is not None:
        an_out["seconds"] = time.time() - t_all
    out = np.clip(x, -1.0, 1.0).astype(np.float32)
    return out[0] if one_d else out


def run(in_path, out_path, mode="voice", opts=None, progress=None):
    if progress:
        progress(0.02, "Loading")
    audio, sr = load_audio(in_path)
    out = process_array(audio, sr, mode, opts, progress)
    save_audio(out_path, out, sr)
    if progress:
        progress(1.0, "Done")
    return out_path
