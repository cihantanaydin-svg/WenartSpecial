"""Deterministic ``.npz`` mesh files (byte-identical for identical geometry → stable CAS hashes)."""

from __future__ import annotations

import io
import zipfile

import numpy as np
from numpy.typing import NDArray

_FIXED_DATE = (1980, 1, 1, 0, 0, 0)


def npz_bytes(arrays: dict[str, NDArray[np.generic]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            arr_buf = io.BytesIO()
            np.lib.format.write_array(
                arr_buf, np.ascontiguousarray(arrays[name]), allow_pickle=False
            )
            zf.writestr(info, arr_buf.getvalue())
    return buf.getvalue()


def mesh_npz(
    vertices: NDArray[np.float64], faces: NDArray[np.int64], uv: NDArray[np.float64]
) -> bytes:
    # Round to 0.01 mm so floating-point noise does not change the hash.
    v = np.round(vertices, 5).astype(np.float32)
    return npz_bytes(
        {"vertices": v, "faces": faces.astype(np.int32), "uv": np.round(uv, 5).astype(np.float32)}
    )
