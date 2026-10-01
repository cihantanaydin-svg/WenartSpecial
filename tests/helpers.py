"""Shared test helpers: synthetic input documents and pipeline drivers."""

from __future__ import annotations

import hashlib
from functools import cache

import numpy as np

from archrender.ingest.uploads import complete_upload, create_upload, put_chunk
from archrender.pipeline.services import Services

MINIMAL_DXF = (
    b"0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n6\n0\nENDSEC\n"
    b"0\nSECTION\n2\nENTITIES\n"
    b"0\nLINE\n8\nDUVAR\n10\n0.0\n20\n0.0\n11\n5.0\n21\n0.0\n"
    b"0\nENDSEC\n0\nEOF\n"
)


@cache
def plan_dxf(seed: int = 3) -> bytes:
    """A synthetic floor plan (walls, doors, windows, room names, dimensions) as a cm DXF, the
    smallest realistic input S1 labels ``floor_plan`` and S2 extracts a valid plan from."""
    from archrender.synth.dxf import plan_dxf as write_dxf
    from archrender.synth.plan import random_spec

    data, _ = write_dxf(random_spec(np.random.default_rng(seed), variant="manhattan"))
    return data


def upload_bytes(
    svc: Services, project_id: str, filename: str, data: bytes, user_id: str = "usr_test"
) -> str:
    sha = hashlib.sha256(data).hexdigest()
    up = create_upload(svc, project_id, filename, len(data), sha, user_id)
    cs = up["chunk_size"]
    for i in range(up["chunks"]):
        chunk = data[i * cs : (i + 1) * cs]
        put_chunk(svc, up["upload_id"], i, chunk, hashlib.sha256(chunk).hexdigest())
    return complete_upload(svc, up["upload_id"], user_id)
