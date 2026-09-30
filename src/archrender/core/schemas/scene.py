"""SceneSpec: the render-ready JSON contract between the app and ``archrender_blender/build.py``.

The Blender side reads this JSON with plain ``json`` (no pydantic inside Blender); the app validates
it here and exports its JSON Schema. Bump ``SCENE_SCHEMA_VERSION`` on any breaking change: both
sides check it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from archrender.core.schemas.common import Strict, Vec3

SCENE_SCHEMA_VERSION = 1

ObjectCategory = Literal[
    "wall",
    "floor",
    "ceiling",
    "skirting",
    "opening_frame",
    "glass",
    "door_leaf",
    "builtin",
    "furniture",
    "decor",
    "cut_cap",
    "backplate",
]

# Categories whose geometry must never be altered by generative refinement.
STRUCTURAL_CATEGORIES: frozenset[str] = frozenset(
    {"wall", "floor", "ceiling", "skirting", "opening_frame", "glass", "door_leaf", "builtin", "cut_cap"}
)

# Stable object pass indices per category (used for masks; 0 = background/sky).
CATEGORY_PASS_INDEX: dict[str, int] = {
    "wall": 1,
    "floor": 2,
    "ceiling": 3,
    "skirting": 4,
    "opening_frame": 5,
    "glass": 6,
    "door_leaf": 7,
    "builtin": 8,
    "furniture": 9,
    "decor": 10,
    "cut_cap": 11,
    "backplate": 12,
}


class MeshFile(Strict):
    """Triangle mesh stored as ``.npz`` with ``vertices`` (N,3 f4), ``faces`` (M,3 i4), ``uv`` (N,2 f4)."""

    path: str  # relative to the scene package directory
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    triangles: int = Field(ge=1)
    watertight: bool


class MaterialSpec(Strict):
    id: str
    base_color_linear: tuple[float, float, float]
    roughness: float = Field(default=0.5, ge=0, le=1)
    metallic: float = Field(default=0.0, ge=0, le=1)
    transmission: float = Field(default=0.0, ge=0, le=1)
    ior: float = Field(default=1.5, ge=1.0, le=3.0)
    alpha: float = Field(default=1.0, ge=0, le=1)
    texture_size_m: tuple[float, float] = (1.0, 1.0)  # real-world size of one texture tile
    base_color_map: str | None = None  # path relative to the package, sRGB
    roughness_map: str | None = None
    normal_map: str | None = None
    pass_index: int = Field(default=0, ge=0, le=255)
    source: str = "procedural"
    license: str = "proprietary-firm"


class SceneObject(Strict):
    id: str
    category: ObjectCategory
    mesh: MeshFile
    material: str
    pass_index: int = Field(ge=0, le=255)
    element_ref: str | None = None  # PlanGraph element id (wall/opening/room)


class SunSpec(Strict):
    azimuth_deg: float  # clockwise from true north
    elevation_deg: float  # apparent elevation, degrees above horizon
    strength: float = Field(default=3.0, ge=0)  # irradiance in W/m², Blender sun strength
    angle_deg: float = Field(default=0.53, ge=0, le=10)
    color_k: float = Field(default=5500.0, ge=1000, le=20000)


class WorldSpec(Strict):
    kind: Literal["color", "sky", "hdri"] = "color"
    color_linear: tuple[float, float, float] = (0.6, 0.7, 0.9)
    strength: float = Field(default=1.0, ge=0)
    hdri_path: str | None = None


class CameraSpec(Strict):
    id: str
    kind: Literal["perspective", "panorama", "orthographic"] = "perspective"
    position: Vec3
    yaw_deg: float  # look direction in plan XY, counter-clockwise from +X
    pitch_deg: float = 0.0  # 0 → two-point perspective (verticals stay vertical)
    focal_mm: float = Field(default=24.0, ge=10, le=120)
    sensor_width_mm: float = 36.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    clip_start: float = Field(default=0.05, gt=0)
    clip_end: float = Field(default=200.0, gt=0)
    ortho_scale: float | None = None


class RenderSettings(Strict):
    engine: Literal["CYCLES"] = "CYCLES"
    device: Literal["AUTO", "CPU", "GPU"] = "AUTO"
    gpu_backends: list[Literal["OPTIX", "CUDA"]] = Field(default_factory=lambda: ["OPTIX", "CUDA"])
    width: int = Field(ge=16, le=8192)
    height: int = Field(ge=16, le=8192)
    samples: int = Field(default=512, ge=1, le=8192)
    adaptive_threshold: float = Field(default=0.01, gt=0)
    denoise: bool = True
    view_transform: Literal["AgX", "Khronos PBR Neutral", "Standard"] = "AgX"
    look: str = "None"
    exposure: float = 0.0
    seed: int = 0
    threads: int = 0  # 0 = auto (CPU)


class ExportSettings(Strict):
    glb: bool = True
    blend: bool = False


class AssetLicense(Strict):
    id: str
    source: str
    license: str
    url: str | None = None
    generated: bool = False


class SceneSpec(Strict):
    schema_version: Literal[1] = SCENE_SCHEMA_VERSION
    scene_id: str
    north_angle_deg: float = 0.0
    objects: list[SceneObject] = Field(min_length=1)
    materials: list[MaterialSpec] = Field(min_length=1)
    sun: SunSpec | None = None
    world: WorldSpec = Field(default_factory=WorldSpec)
    cameras: list[CameraSpec] = Field(default_factory=list)
    render: RenderSettings
    exports: ExportSettings = Field(default_factory=ExportSettings)
    assets: list[AssetLicense] = Field(default_factory=list)
