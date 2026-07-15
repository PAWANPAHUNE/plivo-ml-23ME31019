from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft
from scipy.io import wavfile
from scipy.signal import find_peaks

EPS = 1e-8


def load_wav(path):
    sr, x = wavfile.read(Path(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32) / max(abs(info.min), info.max)
    else:
        x = x.astype(np.float32)
    return x, int(sr)


def _frames(x, frame_length, hop_length):
    if len(x) < frame_length:
        x = np.pad(x, (frame_length - len(x), 0))
    count = 1 + (len(x) - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(count)[:, None]
    return x[idx]


def _slope(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return 0.0
    t = np.linspace(-1.0, 1.0, len(values))
    return float(np.dot(t, values - values.mean()) / (np.dot(t, t) + EPS))


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [0.0] * 5
    return [
        float(values.mean()),
        float(values.std()),
        float(np.percentile(values, 10)),
        float(np.median(values)),
        float(np.percentile(values, 90)),
    ]


def _tail(values, count):
    count = max(1, min(len(values), int(count)))
    return values[-count:]


def _analyse_prefix(prefix, sr):
    hop = max(1, int(round(0.010 * sr)))
    spec_length = max(64, int(round(0.032 * sr)))
    pitch_length = max(80, int(round(0.040 * sr)))
    nfft = 1 << int(np.ceil(np.log2(spec_length)))

    frame = _frames(prefix, spec_length, hop)
    window = np.hanning(spec_length).astype(np.float32)
    rms = np.sqrt(np.mean(frame ** 2, axis=1) + 1e-12)
    energy = 20.0 * np.log10(rms + 1e-8)

    spectrum = np.abs(rfft(frame * window, n=nfft, axis=1)) + 1e-8
    power = spectrum ** 2
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sr)
    total = power.sum(axis=1) + EPS
    centroid = (power * frequencies).sum(axis=1) / total
    bandwidth = np.sqrt(
        (power * (frequencies[None, :] - centroid[:, None]) ** 2).sum(axis=1) / total
    )
    cumulative = np.cumsum(power, axis=1)
    roll_index = (cumulative >= 0.85 * cumulative[:, -1, None]).argmax(axis=1)
    rolloff = frequencies[roll_index]
    flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / (np.mean(spectrum, axis=1) + EPS)
    zcr = np.mean(np.abs(np.diff(np.signbit(frame), axis=1)), axis=1)

    edges = np.array([0, 200, 400, 700, 1000, 1600, 2500, 4000, sr / 2 + 1])
    bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        if mask.any():
            bands.append(np.log10(power[:, mask].mean(axis=1) + 1e-10))
        else:
            bands.append(np.zeros(len(frame)))
    bands = np.stack(bands, axis=1)
    bands -= bands.mean(axis=1, keepdims=True)

    pitch_frame = _frames(prefix, pitch_length, hop)
    pitch_frame = pitch_frame - pitch_frame.mean(axis=1, keepdims=True)
    pitch_frame *= np.hanning(pitch_length)[None, :]
    pitch_nfft = 1 << int(np.ceil(np.log2(2 * pitch_length - 1)))
    autocorrelation = irfft(
        np.abs(rfft(pitch_frame, n=pitch_nfft, axis=1)) ** 2,
        n=pitch_nfft,
        axis=1,
    )[:, :pitch_length]
    autocorrelation /= autocorrelation[:, :1] + EPS
    low_lag = max(1, int(sr / 400.0))
    high_lag = min(pitch_length - 1, int(sr / 60.0))
    lag = low_lag + np.argmax(autocorrelation[:, low_lag:high_lag + 1], axis=1)
    strength = autocorrelation[np.arange(len(autocorrelation)), lag]
    f0 = sr / lag

    return {
        "hop": hop,
        "energy": energy,
        "centroid": centroid / sr,
        "bandwidth": bandwidth / sr,
        "rolloff": rolloff / sr,
        "flatness": flatness,
        "zcr": zcr,
        "bands": bands,
        "f0": f0,
        "pitch_strength": strength,
    }


def extract_features(x, sr, pause_start, pause_index):
    end = max(0, min(len(x), int(round(float(pause_start) * sr))))
    prefix = np.asarray(x[:end], dtype=np.float32).copy()
    if len(prefix) == 0:
        prefix = np.zeros(max(1, int(0.05 * sr)), dtype=np.float32)
    analysis = _analyse_prefix(prefix, sr)
    hop = analysis["hop"]
    recent_count = max(1, int(round(4.0 * sr / hop)))

    energy_all = analysis["energy"]
    energy = _tail(energy_all, recent_count)
    centroid = _tail(analysis["centroid"], recent_count)
    bandwidth = _tail(analysis["bandwidth"], recent_count)
    rolloff = _tail(analysis["rolloff"], recent_count)
    flatness = _tail(analysis["flatness"], recent_count)
    zcr = _tail(analysis["zcr"], recent_count)
    bands = _tail(analysis["bands"], recent_count)

    noise = np.percentile(energy_all, 15)
    high = np.percentile(energy_all, 90)
    threshold = min(high - 12.0, noise + 10.0)
    active = energy > threshold

    index = float(pause_index)
    elapsed = float(pause_start)
    features = [
        index,
        np.log1p(index),
        elapsed,
        np.log1p(elapsed),
        elapsed / (index + 1.0),
        (index + 1.0) / (elapsed + 0.25),
    ]
    features += _stats(energy_all)
    features += [float(high - noise), float(np.mean(energy_all > threshold))]

    for seconds in [0.10, 0.20, 0.35, 0.60, 1.00, 1.50, 2.50, 4.00]:
        count = int(round(seconds * sr / hop))
        values = _tail(energy, count)
        mask = _tail(active, count)
        features += [
            float(values.mean()),
            float(values.std()),
            _slope(values),
            float(mask.mean()),
            float(values.mean() - np.median(energy_all)),
        ]

    for seconds in [0.15, 0.30, 0.50, 0.80]:
        count = max(2, int(round(seconds * sr / hop)))
        current = _tail(energy, count)
        left = max(0, len(energy) - 2 * count)
        right = max(0, len(energy) - count)
        previous = energy[left:right]
        features.append(float(current.mean() - previous.mean()) if len(previous) else 0.0)

    if active.any():
        last = np.where(active)[0][-1]
        cursor = last
        while cursor >= 0 and active[cursor]:
            cursor -= 1
        final_run = last - cursor
        final_gap = len(active) - 1 - last
    else:
        final_run = 0
        final_gap = len(active)

    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0]
    runs = (stops - starts) * hop / sr
    gaps = (starts[1:] - stops[:-1]) * hop / sr if len(starts) > 1 else []
    smooth = np.convolve(energy, np.ones(5) / 5.0, mode="same")
    peaks, _ = find_peaks(
        smooth,
        distance=max(1, int(round(0.12 * sr / hop))),
        prominence=2.0,
    )
    duration = len(energy) * hop / sr
    features += [
        final_run * hop / sr,
        final_gap * hop / sr,
        len(runs) / (duration + EPS),
        len(peaks) / (duration + EPS),
    ]
    features += _stats(runs)
    features += _stats(gaps)

    for values in [centroid, bandwidth, rolloff, flatness, zcr]:
        for seconds in [0.25, 0.60, 1.20]:
            part = _tail(values, int(round(seconds * sr / hop)))
            features += [float(part.mean()), _slope(part)]

    for band in range(bands.shape[1]):
        for seconds in [0.30, 0.80]:
            part = _tail(bands[:, band], int(round(seconds * sr / hop)))
            features += [float(part.mean()), _slope(part)]

    pitch_count = max(1, int(round(2.5 * sr / hop)))
    f0 = _tail(analysis["f0"], pitch_count)
    strength = _tail(analysis["pitch_strength"], pitch_count)
    voiced = (strength > 0.32) & (f0 >= 60.0) & (f0 <= 400.0)
    log_f0 = np.log2(np.clip(f0, 60.0, 400.0))
    features += _stats(log_f0[voiced])
    features += [float(voiced.mean())]

    for seconds in [0.25, 0.50, 1.00, 2.00]:
        count = int(round(seconds * sr / hop))
        voiced_part = _tail(voiced, count)
        pitch_part = _tail(log_f0, count)[voiced_part]
        features += [
            float(pitch_part.mean()) if len(pitch_part) else 0.0,
            _slope(pitch_part),
            float(voiced_part.mean()),
        ]

    if voiced.any():
        last = np.where(voiced)[0][-1]
        cursor = last
        while cursor >= 0 and voiced[cursor]:
            cursor -= 1
        features += [(last - cursor) * hop / sr, (len(voiced) - 1 - last) * hop / sr]
    else:
        features += [0.0, len(voiced) * hop / sr]

    return np.asarray(features, dtype=np.float32)
