# SPDX-License-Identifier: GPL-3.0-or-later
# ArchRender Blender build script. Runs *inside* Blender (official binary or the official bpy wheel)
# as a separate process. It must not import any ArchRender application module: the interface is
# scene.json + mesh files + command-line arguments (ADR-S04, GPL boundary).
#
# Usage:
#   blender -b --factory-startup --python-exit-code 1 --python build.py -- <mode> <package> <out> [camera]
#   python build.py -- <mode> <package> <out> [camera]            (bpy wheel)
# Modes: probe | export | render
"""Build an ArchRender scene in Blender, then probe devices, export GLB/.blend, or render passes."""

import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

SCENE_SCHEMA_VERSION = 1


def log(msg):
    print(f"[archrender-blender] {msg}", flush=True)


def fail(msg, code=2):
    log(f"ERROR: {msg}")
    sys.exit(code)


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        fail("expected arguments after '--'")
    args = argv[argv.index("--") + 1 :]
    if not args:
        fail("missing mode")
    return args


# --------------------------------------------------------------------------------------------
# devices
# --------------------------------------------------------------------------------------------
def configure_device(scene, render):
    """Pick OPTIX → CUDA (per the preference list) or CPU. Returns a dict describing the choice."""
    requested = render.get("device", "AUTO")
    info = {"requested": requested, "device": "CPU", "backend": "CPU", "devices": []}
    if requested in ("AUTO", "GPU"):
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in render.get("gpu_backends", ["OPTIX", "CUDA"]):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue  # backend not compiled in / driver libraries missing
            prefs.refresh_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                info.update(device="GPU", backend=backend, devices=[d.name for d in gpus])
                break
        if info["device"] == "CPU" and requested == "GPU":
            fail("GPU rendering requested but no OPTIX/CUDA device is available", code=3)
    if info["device"] == "CPU":
        scene.cycles.device = "CPU"
        threads = int(render.get("threads", 0))
        if threads > 0:
            scene.render.threads_mode = "FIXED"
            scene.render.threads = threads
    return info


# --------------------------------------------------------------------------------------------
# scene construction
# --------------------------------------------------------------------------------------------
def load_mesh(package, mesh_ref, name):
    path = os.path.join(package, mesh_ref["path"])
    data = np.load(path)
    verts = data["vertices"].astype(np.float64)
    faces = data["faces"].astype(np.int64)
    uv = data["uv"].astype(np.float64) if "uv" in data else None
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    if uv is not None:
        layer = mesh.uv_layers.new(name="UVMap")
        # vertices are split per corner, so loop i uses vertex i and uv[i]
        loop_vidx = np.empty(len(mesh.loops), dtype=np.int64)
        mesh.loops.foreach_get("vertex_index", loop_vidx)
        layer.data.foreach_set("uv", uv[loop_vidx].astype(np.float32).ravel())
    mesh.validate(clean_customdata=False)
    mesh.update()
    return mesh


def principled(mat):
    if mat.node_tree is None:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        out = nodes.get("Material Output") or nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return bsdf


def image_texture(mat, package, rel_path, non_color, size_m):
    nt = mat.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(package, rel_path), check_existing=True)
    if non_color:
        tex.image.colorspace_settings.name = "Non-Color"
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0 / size_m[0], 1.0 / size_m[1], 1.0)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    return tex


def build_material(spec, package):
    mat = bpy.data.materials.new(spec["id"])
    bsdf = principled(mat)
    r, g, b = spec["base_color_linear"]
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = spec.get("roughness", 0.5)
    bsdf.inputs["Metallic"].default_value = spec.get("metallic", 0.0)
    bsdf.inputs["IOR"].default_value = spec.get("ior", 1.5)
    bsdf.inputs["Alpha"].default_value = spec.get("alpha", 1.0)
    transmission = spec.get("transmission", 0.0)
    bsdf.inputs["Transmission Weight"].default_value = transmission
    if transmission > 0.5 and "Thin Wall" in bsdf.inputs:
        bsdf.inputs["Thin Wall"].default_value = True  # architectural glazing pane
    size = spec.get("texture_size_m", [1.0, 1.0])
    nt = mat.node_tree
    if spec.get("base_color_map"):
        tex = image_texture(mat, package, spec["base_color_map"], False, size)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if spec.get("roughness_map"):
        tex = image_texture(mat, package, spec["roughness_map"], True, size)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Roughness"])
    if spec.get("normal_map"):
        tex = image_texture(mat, package, spec["normal_map"], True, size)
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    mat.pass_index = int(spec.get("pass_index", 0))
    return mat


def sun_vector(azimuth_deg, elevation_deg, north_angle_deg):
    """Unit vector toward the sun in the plan frame (plan +Y rotated from true north)."""
    plan_az = math.radians(north_angle_deg + azimuth_deg)
    el = math.radians(elevation_deg)
    return Vector((math.sin(plan_az) * math.cos(el), math.cos(plan_az) * math.cos(el), math.sin(el)))


def build_scene(spec, package):
    if spec.get("schema_version") != SCENE_SCHEMA_VERSION:
        fail(f"scene schema_version {spec.get('schema_version')} != {SCENE_SCHEMA_VERSION}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    materials = {m["id"]: build_material(m, package) for m in spec["materials"]}
    for obj_spec in spec["objects"]:
        mesh = load_mesh(package, obj_spec["mesh"], obj_spec["id"])
        mat = materials.get(obj_spec["material"])
        if mat is None:
            fail(f"object {obj_spec['id']} references unknown material {obj_spec['material']}")
        mesh.materials.append(mat)
        obj = bpy.data.objects.new(obj_spec["id"], mesh)
        obj.pass_index = int(obj_spec["pass_index"])
        obj["archrender_category"] = obj_spec["category"]
        if obj_spec.get("element_ref"):
            obj["archrender_element"] = obj_spec["element_ref"]
        scene.collection.objects.link(obj)

    sun = spec.get("sun")
    if sun:
        light = bpy.data.lights.new("sun", "SUN")
        light.energy = sun["strength"]
        light.angle = math.radians(sun["angle_deg"])
        light.use_temperature = True
        light.temperature = sun["color_k"]
        obj = bpy.data.objects.new("sun", light)
        obj.rotation_euler = sun_vector(
            sun["azimuth_deg"], sun["elevation_deg"], spec.get("north_angle_deg", 0.0)
        ).to_track_quat("Z", "Y").to_euler()
        scene.collection.objects.link(obj)

    world = bpy.data.worlds.new("world")
    scene.world = world
    wspec = spec.get("world", {})
    if world.node_tree is None:
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    out = world.node_tree.nodes.get("World Output") or world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
    kind = wspec.get("kind", "color")
    if kind == "hdri" and wspec.get("hdri_path"):
        env = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(os.path.join(package, wspec["hdri_path"]))
        world.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
    else:
        r, g, b = wspec.get("color_linear", [0.6, 0.7, 0.9])
        bg.inputs["Color"].default_value = (r, g, b, 1.0)
    bg.inputs["Strength"].default_value = wspec.get("strength", 1.0)
    return scene


def add_camera(scene, cam_spec, render):
    cam = bpy.data.cameras.new(cam_spec["id"])
    kind = cam_spec.get("kind", "perspective")
    if kind == "orthographic":
        cam.type = "ORTHO"
        cam.ortho_scale = cam_spec.get("ortho_scale") or 10.0
    elif kind == "panorama":
        cam.type = "PANO"
        cam.panorama_type = "EQUIRECTANGULAR"
    else:
        cam.type = "PERSP"
        cam.lens = cam_spec["focal_mm"]
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = cam_spec.get("sensor_width_mm", 36.0)
    cam.shift_x = cam_spec.get("shift_x", 0.0)
    cam.shift_y = cam_spec.get("shift_y", 0.0)
    cam.clip_start = cam_spec.get("clip_start", 0.05)
    cam.clip_end = cam_spec.get("clip_end", 200.0)
    obj = bpy.data.objects.new(cam_spec["id"], cam)
    p = cam_spec["position"]
    obj.location = (p["x"], p["y"], p["z"])
    # Blender cameras look down local -Z with +Y up: pitch 0 → rotate X by 90°; yaw about Z.
    obj.rotation_euler = (
        math.radians(90.0 + cam_spec.get("pitch_deg", 0.0)),
        0.0,
        math.radians(cam_spec["yaw_deg"] - 90.0),
    )
    scene.collection.objects.link(obj)
    scene.camera = obj
    return obj


def configure_render(scene, render):
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = render["width"]
    scene.render.resolution_y = render["height"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.cycles.samples = render["samples"]
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = render["adaptive_threshold"]
    scene.cycles.seed = render.get("seed", 0)
    scene.cycles.use_denoising = bool(render.get("denoise", True))
    if render.get("denoise", True):
        try:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        except TypeError:
            log("OpenImageDenoise unavailable; using the default denoiser")
    scene.render.film_transparent = False
    scene.view_settings.view_transform = render.get("view_transform", "AgX")
    scene.view_settings.look = render.get("look", "None")
    scene.view_settings.exposure = render.get("exposure", 0.0)
    vl = scene.view_layers[0]
    vl.use_pass_combined = True
    vl.use_pass_z = True
    vl.use_pass_normal = True
    vl.use_pass_object_index = True
    vl.use_pass_material_index = True
    vl.use_pass_diffuse_color = True


# --------------------------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------------------------
def mode_probe(out_dir):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    info = configure_device(scene, {"device": "AUTO", "gpu_backends": ["OPTIX", "CUDA"]})
    info["blender"] = bpy.app.version_string
    write_json(os.path.join(out_dir, "probe.json"), info)


def mode_export(spec, package, out_dir):
    build_scene(spec, package)
    exports = spec.get("exports", {})
    result = {"blender": bpy.app.version_string, "files": []}
    if exports.get("glb", True):
        path = os.path.join(out_dir, "scene.glb")
        bpy.ops.export_scene.gltf(
            filepath=path, export_format="GLB", export_cameras=False, export_lights=False
        )
        result["files"].append("scene.glb")
    if exports.get("blend", False):
        path = os.path.join(out_dir, "scene.blend")
        bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
        result["files"].append("scene.blend")
    write_json(os.path.join(out_dir, "export.json"), result)


def mode_render(spec, package, out_dir, camera_id):
    scene = build_scene(spec, package)
    cams = [c for c in spec.get("cameras", []) if c["id"] == camera_id]
    if not cams:
        fail(f"camera {camera_id} not in scene")
    render = spec["render"]
    add_camera(scene, cams[0], render)
    configure_render(scene, render)
    device = configure_device(scene, render)
    settings = scene.render.image_settings
    settings.media_type = "MULTI_LAYER_IMAGE"
    settings.file_format = "OPEN_EXR_MULTILAYER"
    settings.color_depth = "32"
    settings.exr_codec = "ZIP"
    scene.render.filepath = os.path.join(out_dir, "passes.exr")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    seconds = time.time() - t0
    # view-transformed beauty (AgX) as 16-bit PNG
    settings.media_type = "IMAGE"
    settings.file_format = "PNG"
    settings.color_mode = "RGB"
    settings.color_depth = "16"
    settings.compression = 15
    bpy.data.images["Render Result"].save_render(os.path.join(out_dir, "beauty.png"), scene=scene)
    write_json(
        os.path.join(out_dir, "render.json"),
        {
            "blender": bpy.app.version_string,
            "device": device,
            "seconds": round(seconds, 3),
            "samples": render["samples"],
            "width": render["width"],
            "height": render["height"],
            "view_transform": scene.view_settings.view_transform,
            "camera": camera_id,
            "objects": {o.name: o.pass_index for o in scene.objects if o.type == "MESH"},
        },
    )


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def main():
    args = parse_args()
    mode = args[0]
    if mode == "probe":
        if len(args) < 2:
            fail("usage: probe <out_dir>")
        os.makedirs(args[1], exist_ok=True)
        mode_probe(args[1])
        return
    if len(args) < 3:
        fail("usage: <export|render> <package_dir> <out_dir> [camera_id]")
    package, out_dir = args[1], args[2]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(package, "scene.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    if mode == "export":
        mode_export(spec, package, out_dir)
    elif mode == "render":
        if len(args) < 4:
            fail("render needs a camera id")
        mode_render(spec, package, out_dir, args[3])
    else:
        fail(f"unknown mode {mode}")
    log("done")


if __name__ == "__main__":
    main()
