"""Connectivity features for a prepared samples-by-seven network time series."""

import numpy as np

EDGES = np.triu_indices(7, 1)


def _field(timeseries):
    values = np.asarray(timeseries, dtype=float)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("Expected a finite samples-by-seven network time series")
    if len(values) < 2:
        raise ValueError("At least two samples are required")
    return values


def correlation(timeseries):
    """Pearson correlations without a Fisher transform."""
    values = _field(timeseries)
    centered = values - values.mean(axis=0)
    scales = np.sqrt((centered * centered).sum(axis=0))
    if np.any(scales <= 0):
        raise ValueError("Correlation is undefined for a constant network time series")
    return centered.T @ centered / np.outer(scales, scales)


def window_dispersion(timeseries, width, starts):
    """Sample standard deviation of the 21 correlations across selected windows."""
    values = _field(timeseries)
    starts = list(starts)
    if (not isinstance(width, (int, np.integer)) or width < 2 or len(starts) < 2
            or any(not isinstance(start, (int, np.integer))
                   or start < 0 or start + width > len(values) for start in starts)):
        raise ValueError("Dispersion requires at least two complete windows")
    correlations = np.stack([correlation(values[start:start + width])[EDGES] for start in starts])
    return correlations.std(axis=0, ddof=1)


def extract(timeseries, tr_seconds):
    """Extract static FC and dispersion after preprocessing and M24 regression."""
    values = _field(timeseries)
    if not np.isfinite(tr_seconds) or tr_seconds <= 0:
        raise ValueError("TR must be positive and measured in seconds")
    width = max(8, round(60 / tr_seconds))
    stride = max(1, round(30 / tr_seconds))
    starts = list(range(0, len(values) - width + 1, stride))
    return {
        "static": correlation(values)[EDGES],
        "dispersion": window_dispersion(values, width, starts),
        "window_samples": width,
        "stride_samples": stride,
        "starts": starts,
    }


def common_phase(timeseries, seed):
    """Randomize Fourier phase jointly across networks, preserving cross-spectra."""
    values = _field(timeseries)
    fourier = np.fft.rfft(values, axis=0)
    rng = np.random.default_rng(seed)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, len(fourier)))
    phases[0] = 1
    if len(values) % 2 == 0:
        phases[-1] = rng.choice([-1.0, 1.0])
    return np.fft.irfft(fourier * phases[:, None], n=len(values), axis=0)


def nonoverlapping_windows(samples, width):
    """Starting indices for the four-window temporal-coverage controls."""
    if (not isinstance(samples, (int, np.integer))
            or not isinstance(width, (int, np.integer)) or width < 2 or samples < 4 * width):
        raise ValueError("The record must support four complete nonoverlapping windows")
    starts = {}
    for name, offset in (("first4", 0), ("middle4", (samples - 4 * width) // 2),
                         ("last4", samples - 4 * width)):
        starts[name] = (np.arange(4) * width + offset).tolist()
    starts["spread4"] = np.floor(np.linspace(0, samples - width, 4)).astype(int).tolist()
    return starts


def window_indices(count, mode="full"):
    """Select all windows or eight compact/distributed windows."""
    if not isinstance(count, (int, np.integer)) or count < 8:
        raise ValueError("At least eight windows are required")
    if mode == "full":
        return np.arange(count)
    if mode == "first8":
        return np.arange(8)
    if mode == "middle8":
        start = (count - 8) // 2
        return np.arange(start, start + 8)
    if mode == "last8":
        return np.arange(count - 8, count)
    if mode == "spread8":
        return np.arange(8) * (count - 1) // 7
    raise ValueError(f"Unknown window selection: {mode}")
