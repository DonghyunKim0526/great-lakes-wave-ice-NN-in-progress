"""Executable WIFF1.0 components, ported from the authors' MIT-licensed v1.0b.

The original MATLAB and Fortran remain in vendor/wiff-v1.0b for comparison.
This module implements the local fracture kernel, not CICE FSD time evolution.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import maximum_filter1d, minimum_filter1d

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "wiff-v1.0b"


def parameters():
    values = loadmat(VENDOR / "Misc/NN_params.mat")
    f = values["f"].ravel().astype(float)
    edges = values["redge"].ravel().astype(float)
    centers = values["rcent"].ravel().astype(float)
    return f, f * (np.sqrt(1.1) - np.sqrt(1 / 1.1)), edges, centers


def softmax(logits):
    values = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


class DenseNetwork:
    """Exact layer dimensions and raw-input convention of the archived networks."""

    def __init__(self, kind="histogram", source="text"):
        if kind not in {"histogram", "classifier"}:
            raise ValueError("kind must be histogram or classifier")
        self.kind = kind
        widths = [27, 100, 100, 100, 100, 100, 12] if kind == "histogram" else [27, 100, 100, 2]
        stem = VENDOR / ("Model/NN_v1" if kind == "histogram" else "Classifier/Classifier_v1")
        self.layers = []
        if source == "text":
            values = np.fromstring(stem.with_suffix(".txt").read_text(), sep=",")
            offset = 0
            for left, right in zip(widths[:-1], widths[1:]):
                count = left * right
                kernel = values[offset:offset + count].reshape(left, right)
                offset += count
                bias = values[offset:offset + right]
                offset += right
                self.layers.append((kernel, bias))
            if offset != len(values):
                raise ValueError("Unexpected weight file length")
        elif source == "hdf5":
            import h5py, json
            with h5py.File(stem.with_suffix(".h5"), "r") as handle:
                config = json.loads(handle.attrs["model_config"])
                layers = config["config"]
                if isinstance(layers, dict):
                    layers = layers["layers"]
                for layer in layers:
                    if layer["class_name"] != "Dense":
                        continue
                    name = layer["config"]["name"]
                    group = handle["model_weights"][name]
                    arrays = {}
                    def collect(key, value):
                        if isinstance(value, h5py.Dataset):
                            arrays[key.split("/")[-1].split(":")[0]] = value[()].astype(float)
                    group.visititems(collect)
                    self.layers.append((arrays["kernel"], arrays["bias"]))
        else:
            raise ValueError("source must be text or hdf5")
        if [(k.shape[0], k.shape[1]) for k, _ in self.layers] != list(zip(widths[:-1], widths[1:])):
            raise ValueError("Network dimensions do not match the archive")

    def predict(self, inputs, batch_size=4096):
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        if inputs.ndim != 2 or inputs.shape[1] != 27 or not np.isfinite(inputs).all():
            raise ValueError("Expected finite inputs with shape (N, 27)")
        result = []
        for start in range(0, len(inputs), batch_size):
            value = inputs[start:start + batch_size]
            for i, (kernel, bias) in enumerate(self.layers):
                value = value @ kernel + bias
                value = softmax(value) if i == len(self.layers) - 1 else np.maximum(value, 0)
            result.append(value)
        return np.concatenate(result) if result else np.empty((0, self.layers[-1][1].size))


def eligible(inputs):
    """Paper's input screening; the stand-alone SP kernel itself does not use C."""
    x = np.atleast_2d(inputs)
    _, df, _, _ = parameters()
    hs = 4 * np.sqrt(np.maximum(0, x[:, :25] @ df))
    return (x[:, 25] < 10) & (x[:, 25] >= 0) & (x[:, 26] > .01) & (hs > .1)


def predict_gated(inputs, histogram=None, classifier=None, threshold=.54):
    """Follow the archived Fortran: run iff P(fracture) >= 0.54 after screening."""
    x = np.atleast_2d(inputs)
    histogram = histogram or DenseNetwork()
    classifier = classifier or DenseNetwork("classifier")
    active = eligible(x)
    probability = np.zeros(len(x))
    probability[active] = classifier.predict(x[active])[:, 1]
    run = active & (probability >= threshold)
    output = np.zeros((len(x), 12))
    output[run] = histogram.predict(x[run])
    return output, probability, run


def peakfinder(eta, spacing=10):
    """Port peakfinder2.m, including insertion of missing alternating extrema."""
    eta = np.asarray(eta)
    size = 2 * spacing + 1
    maxima = np.flatnonzero(eta >= maximum_filter1d(eta, size=size, mode="nearest"))
    minima = np.flatnonzero(eta <= minimum_filter1d(eta, size=size, mode="nearest"))
    for left, right in zip(minima[:-1], minima[1:]):
        if right > left + 1 and not np.any((maxima > left) & (maxima < right)):
            maxima = np.sort(np.append(maxima, left + 1 + np.argmax(eta[left + 1:right])))
    for left, right in zip(maxima[:-1], maxima[1:]):
        if right > left + 1 and not np.any((minima > left) & (minima < right)):
            minima = np.sort(np.append(minima, left + 1 + np.argmin(eta[left + 1:right])))
    return maxima, minima


@dataclass
class FractureResult:
    histogram: np.ndarray
    iterations: int
    converged: bool
    changes: np.ndarray
    histograms: np.ndarray
    fracture_count: int


class SPWIFF:
    """NumPy port of archived SP_WIFF_Standalone.m, with reproducible phases.

    Defaults match the released stand-alone script: 10 km, dx=1 m,
    +/-10 m peak window, strain threshold 3e-5, tolerance 6.5e-4.
    The paper states 5e-4; use tolerance=5e-4 to test that setting.
    """
    def __init__(self, domain=10_000, dx=1, strain_threshold=3e-5):
        self.f, self.df, self.edges, self.centers = parameters()
        self.x = np.arange(dx, domain + dx / 2, dx)
        self.spacing = int(np.floor(10 / dx + .5))
        self.strain_threshold = strain_threshold
        wavelength = 9.80616 / (2 * np.pi * self.f ** 2)
        angle = (2 * np.pi / wavelength[:, None]) * self.x[None, :]
        self.cosines, self.sines = np.cos(angle), np.sin(angle)

    def fracture_lengths(self, spectrum, thickness, phases):
        coefficients = np.sqrt(2 * np.asarray(spectrum) * self.df)
        eta = (coefficients * np.cos(phases)) @ self.cosines - (coefficients * np.sin(phases)) @ self.sines
        maxima, minima = peakfinder(eta, self.spacing)
        locs = np.unique(np.concatenate((maxima, minima)))
        x, y = self.x[locs], eta[locs]
        if len(locs) < 3:
            return np.empty(0)
        dx1, dx2 = np.diff(x)[:-1], np.diff(x)[1:]
        curvature = 2 * (y[:-2] * dx2 - y[1:-1] * (dx1 + dx2) + y[2:] * dx1) / (dx1 * dx2 * (dx1 + dx2))
        strain = np.zeros(len(x))
        strain[1:-1] = .5 * thickness * np.abs(curvature)
        return np.diff(x[strain > self.strain_threshold]) / 2

    def run(self, spectrum, thickness, seed=20261024, max_iterations=1000, tolerance=6.5e-4, phases=None):
        spectrum = np.asarray(spectrum, dtype=float)
        if spectrum.shape != (25,) or np.any(spectrum < 0) or not np.isfinite(spectrum).all() or thickness < 0:
            raise ValueError("Expected a nonnegative 25-bin spectrum and thickness")
        if max_iterations < 1 or (phases is not None and max_iterations != 1):
            raise ValueError("Fixed phases require exactly one iteration")
        rng = np.random.default_rng(seed)
        counts = np.zeros(12)
        previous = np.zeros(12)
        histories, changes = [], []
        fracture_count = 0
        for i in range(max_iterations):
            random_phase = rng.uniform(0, 2*np.pi, 25) if phases is None else np.asarray(phases)
            lengths = self.fracture_lengths(spectrum, thickness, random_phase)
            fracture_count += len(lengths)
            counts += np.histogram(lengths, self.edges)[0]
            counts[-1] += np.count_nonzero(lengths > self.edges[-1])
            weighted = counts * self.centers
            histogram = weighted / weighted.sum() if weighted.sum() else weighted.copy()
            change = np.mean(np.abs(histogram - previous))
            histories.append(histogram.copy()); changes.append(change)
            previous = histogram
            if phases is None and i >= 1 and change <= tolerance:
                break
        return FractureResult(histogram, i+1, bool(phases is None and i>=1 and change<=tolerance), np.asarray(changes), np.asarray(histories), fracture_count)


def histogram_metrics(true, predicted, centers=None):
    """Mass-vector form of paper Eq. 6-9: bin widths are already in q=A_i dr_i."""
    centers = parameters()[3] if centers is None else np.asarray(centers)
    true, predicted = np.asarray(true), np.asarray(predicted)
    radius = true @ centers
    rse = np.sum(np.abs(true - predicted) * centers, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sse = rse / radius
        symmetric_radius_error = np.abs(predicted @ centers - radius) / (.5 * (predicted @ centers + radius))
    return {"radius": radius, "rse_m": rse, "sse": sse, "sae": np.abs(true-predicted).sum(axis=-1), "symmetric_radius_error": symmetric_radius_error}
