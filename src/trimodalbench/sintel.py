"""Reader for the MPI Sintel training archives."""

from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path
import re
import struct

import numpy as np
from PIL import Image

from .geometry.camera import Camera


def _payload(path: Path) -> bytes:
    data = path.read_bytes()
    if data[:4] != b"PIEH":
        raise ValueError(f"Invalid Sintel magic in {path}")
    return data[4:]


def read_camera(path: Path) -> Camera:
    data = _payload(path)
    if len(data) != 21 * 8:
        raise ValueError(f"Invalid camera record size in {path}")
    values = np.frombuffer(data, dtype="<f8")
    world_to_camera = np.eye(4)
    world_to_camera[:3] = values[9:].reshape(3, 4)
    K = values[:9].reshape(3, 3)
    # The stored extrinsic is the world-to-camera term in P = K [R | t].
    return Camera(K, np.linalg.inv(world_to_camera))


def _read_dense(path: Path, channels: int) -> np.ndarray:
    data = _payload(path)
    if len(data) < 8:
        raise ValueError(f"Missing image dimensions in {path}")
    width, height = struct.unpack_from("<ii", data)
    if width <= 0 or height <= 0 or len(data) != 8 + 4 * width * height * channels:
        raise ValueError(f"Invalid dimensions or payload length in {path}")
    shape = (height, width) if channels == 1 else (height, width, channels)
    return np.frombuffer(data, dtype="<f4", offset=8).reshape(shape)


def _image(path: Path, mode: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert(mode))


@dataclass(frozen=True)
class FramePair:
    source_id: int
    target_id: int
    source_rgb: np.ndarray
    target_rgb: np.ndarray
    source_depth: np.ndarray
    target_depth: np.ndarray
    source_camera: Camera
    target_camera: Camera
    flow: np.ndarray
    flow_valid: np.ndarray
    visible: np.ndarray
    static: np.ndarray | None


def iter_frame_pairs(
    training: Path, sequence: str, *, start: int | None = None,
    count: int | None = None, static_dir: Path | None = None,
) -> Iterator[FramePair]:
    """Align by original Sintel frame ID. The flow filename names its source."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", sequence):
        raise ValueError("Expected a Sintel sequence name")
    if count is not None and count < 1:
        raise ValueError("Count must be positive")
    clean = training / "clean" / sequence
    frames = sorted(
        int(match[1]) for path in clean.glob("*.png")
        if (match := re.fullmatch(r"frame_(\d{4})\.png", path.name))
    )
    start = frames[0] if frames and start is None else start
    if start not in frames:
        raise ValueError(f"INVALID_ALIGNMENT: missing source frame in {clean}")
    stop = frames[-1] if count is None else start + count
    if stop <= start:
        raise ValueError("INVALID_ALIGNMENT: no adjacent frames")
    frame_ids = set(frames)

    def path(modality: str, frame: int, suffix: str) -> Path:
        return training / modality / sequence / f"frame_{frame:04d}.{suffix}"

    for source in range(start, stop):
        target = source + 1
        if source not in frame_ids or target not in frame_ids:
            raise ValueError(f"INVALID_ALIGNMENT: missing RGB in pair {source}->{target}")
        try:
            source_rgb, target_rgb = (_image(path("clean", i, "png"), "RGB") for i in (source, target))
            source_depth, target_depth = (_read_dense(path("depth", i, "dpt"), 1) for i in (source, target))
            source_camera, target_camera = (read_camera(path("camdata_left", i, "cam")) for i in (source, target))
            flow = _read_dense(path("flow", source, "flo"), 2)
            flow_valid = _image(path("invalid", source, "png"), "L") == 0
            visible = _image(path("occlusions", source, "png"), "L") == 0
            static = None if static_dir is None else _image(
                static_dir / f"frame_{source:04d}.png", "L"
            ) != 0
        except FileNotFoundError as exc:
            raise ValueError(f"INVALID_ALIGNMENT: missing {exc.filename} for {source}->{target}") from exc
        shape = source_rgb.shape[:2]
        if any(
            array.shape[:2] != shape for array in
            (target_rgb, source_depth, target_depth, flow, flow_valid, visible, static)
            if array is not None
        ):
            raise ValueError(f"INVALID_ALIGNMENT: inconsistent image dimensions for {source}->{target}")
        yield FramePair(source, target, source_rgb, target_rgb, source_depth, target_depth,
                        source_camera, target_camera, flow, flow_valid, visible, static)
