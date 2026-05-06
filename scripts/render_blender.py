from __future__ import annotations

"""Render MuJoCo scenes and rollout logs through Blender.

Run inside Blender, for example:

blender --background --python scripts/render_blender.py -- \
  --source logs/closed_loop_lerobot/20260427_120000_task_name

The script supports:
- a MuJoCo scene XML or JSON snapshot
- a rollout log directory containing model.mjb, states.npy.zst, and info.json

It builds a Blender scene from the MuJoCo bodies/geoms, optionally animates
the bodies from qpos, and renders the result with Blender's render engine.
"""

from contextlib import contextmanager
from io import BytesIO
import json
import os
import subprocess
import site
from pathlib import Path
import re
import tempfile
from typing import Any

import bpy
for _python_path in str(os.environ.get("PYTHONPATH", "")).split(os.pathsep):
    if _python_path:
        site.addsitedir(_python_path)
import mujoco
import numpy as np
import trimesh
import zstandard as zstd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets"


def _try_load_optional_mjlab_plugin() -> None:
    loader = getattr(mujoco, "mj_loadPluginLibrary", None)
    if not callable(loader):
        return
    candidates: list[str] = []
    env_candidate = str(os.environ.get("MJLAB_PLUGIN_PATH", "")).strip()
    if env_candidate:
        candidates.append(env_candidate)
    candidates.extend([
        str(PROJECT_ROOT / "libmjlab.so.3.3.0"),
        "./libmjlab.so.3.3.0",
        "./libmjlab.so",
        "./libmjlab.dylib",
        "./mjlab.dll",
    ])
    for plugin_path in candidates:
        try:
            loader(plugin_path)
            return
        except Exception:
            continue


_try_load_optional_mjlab_plugin()


@contextmanager
def temporary_active_collection(collection: bpy.types.Collection):
    view_layer = bpy.context.view_layer
    previous = view_layer.active_layer_collection
    view_layer.active_layer_collection = view_layer.layer_collection.children[collection.name]
    try:
        yield
    finally:
        view_layer.active_layer_collection = previous


def ensure_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def ensure_material(name: str, rgba: np.ndarray) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        alpha = float(rgba[3]) if len(rgba) > 3 else 1.0
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            linear_rgb = srgb_to_linear(tuple(float(x) for x in rgba[:3]))
            bsdf.inputs["Base Color"].default_value = (*linear_rgb, 1.0)
            bsdf.inputs["Alpha"].default_value = alpha
        material.diffuse_color = (*srgb_to_linear(tuple(float(x) for x in rgba[:3])), alpha)
        if alpha < 1.0:
            material.blend_method = "BLEND"
            material.shadow_method = "HASHED"
    return material


def srgb_to_linearrgb(c: float) -> float:
    if c < 0.04045:
        return 0.0 if c < 0.0 else c * (1.0 / 12.92)
    return pow((c + 0.055) * (1.0 / 1.055), 2.4)


def srgb_to_linear(c: tuple[float, ...]) -> tuple[float, ...]:
    assert len(c) == 3
    return tuple(srgb_to_linearrgb(x) for x in c)


def add_mesh(name: str, vertices: np.ndarray, faces: np.ndarray) -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices=vertices, edges=[], faces=faces, shade_flat=True)
    mesh.validate()
    mesh.materials.append(None)
    return mesh


def import_obj_mesh(path: Path) -> None:
    bpy.ops.wm.obj_import(
        filepath=str(path),
        forward_axis="Y",
        up_axis="Z",
        use_split_objects=False,
        use_split_groups=False,
    )


def import_stl_mesh(path: Path) -> None:
    bpy.ops.wm.stl_import(
        filepath=str(path),
        forward_axis="Y",
        up_axis="Z",
    )


def import_mesh(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        import_obj_mesh(path)
    elif suffix == ".stl":
        import_stl_mesh(path)
    else:
        raise NotImplementedError(f"Unsupported mesh format: {path}")


def add_fixed_camera(name: str, fovy: float, clip_start: float, clip_end: float) -> bpy.types.Camera:
    camera = bpy.data.cameras.new(name)
    camera.lens_unit = "FOV"
    camera.sensor_fit = "VERTICAL"
    camera.angle = fovy
    camera.clip_start = clip_start
    camera.clip_end = clip_end
    return camera


def add_object(
    name: str,
    data: bpy.types.Mesh | bpy.types.Camera | None = None,
    parent: bpy.types.Object | None = None,
    collection: bpy.types.Collection | None = None,
    location: np.ndarray | None = None,
    rotation_quaternion: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    material: bpy.types.Material | None = None,
    *,
    allow_rename: bool = False,
) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, data)
    if not allow_rename and name != obj.name:
        raise ValueError(f"Invalid object name: {name} != {obj.name}")
    obj.rotation_mode = "QUATERNION"
    if collection is None:
        collection = bpy.context.scene.collection
    collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    if location is not None:
        obj.location = location
    if rotation_quaternion is not None:
        obj.rotation_quaternion = rotation_quaternion
    if scale is not None:
        obj.scale = scale
    if material is not None and obj.type == "MESH":
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = material
    return obj


def parse_float_list(text: str | None, default: list[float]) -> list[float]:
    if not text:
        return list(default)
    try:
        return [float(x) for x in text.strip().split()]
    except Exception:
        return list(default)


def format_float_list(values) -> str:
    return " ".join(f"{float(x):.6f}" for x in values)


def sanitize_key(text: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    return safe.strip("_") or "asset"


def relpath_posix(target: Path, base: Path) -> str:
    try:
        return Path(target).resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return Path(os.path.relpath(str(target.resolve()), str(base.resolve()))).as_posix()


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    n2 = float(np.dot(q, q))
    if n2 < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float) / n2


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    out = quat_mul(quat_mul(q, qv), quat_inv(q))
    return out[1:]


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    mujoco.mju_axisAngle2Quat(out, axis, angle)
    return out


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    qx = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), roll)
    qy = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), pitch)
    qz = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)
    q = quat_mul(qz, quat_mul(qy, qx))
    return q / np.linalg.norm(q)


def mat9_to_quat_wxyz(mat9: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.asarray(mat9, dtype=float))
    n = float(np.linalg.norm(quat))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / n


def quat_to_euler_rpy_deg(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(float(sinp)) >= 1.0:
        pitch = np.pi / 2.0 * np.sign(sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=float))


def wrap_deg180(angles_deg: np.ndarray) -> np.ndarray:
    vals = np.asarray(angles_deg, dtype=float)
    return (vals + 180.0) % 360.0 - 180.0


def match_and_remove_prefix(text: str, prefixes: list[str]) -> str | None:
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return None


def remove_prefix(text: str, prefix: str):
    if not text.startswith(prefix):
        raise ValueError(f"Invalid prefix: {text} does not start with {prefix}")
    return text[len(prefix) :]


def parse_identifier(body_name: str):
    if ":" in body_name:
        prefix, identifier = body_name.split(":", 1)
        if "/" not in prefix:
            return None
        prefix, namespace = prefix.rsplit("/", 1)
        identifier = f"{namespace}:{identifier}"
        parts = identifier.split(":")
        return ":".join(parts[-2:])
    if "/" not in body_name:
        return None
    _, identifier = body_name.rsplit("/", 1)
    return identifier


def compute_relative_transform(pos: np.ndarray, quat: np.ndarray, parent_pos: np.ndarray, parent_quat: np.ndarray):
    assert pos.ndim == quat.ndim
    assert pos.shape[-1] == 3 and pos.shape == parent_pos.shape
    assert quat.shape[-1] == 4 and quat.shape == parent_quat.shape
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_quat(quat, scalar_first=True)
    parent_rotation = Rotation.from_quat(parent_quat, scalar_first=True)

    relative_pos = parent_rotation.inv().apply(pos - parent_pos)
    relative_rotation = parent_rotation.inv() * rotation
    relative_quat = relative_rotation.as_quat(scalar_first=True)
    return relative_pos, relative_quat


def compute_parent_transform(pos: np.ndarray, quat: np.ndarray, relative_pos: np.ndarray, relative_quat: np.ndarray):
    assert pos.ndim == quat.ndim
    assert pos.shape[-1] == 3 and pos.shape == relative_pos.shape
    assert quat.shape[-1] == 4 and quat.shape == relative_quat.shape
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_quat(quat, scalar_first=True)
    relative_rotation = Rotation.from_quat(relative_quat, scalar_first=True)

    parent_rotation = rotation * relative_rotation.inv()
    parent_pos = pos - parent_rotation.apply(relative_pos)
    parent_quat = parent_rotation.as_quat(scalar_first=True)
    return parent_pos, parent_quat


def parse_scene_json(scene_json_path: Path) -> dict[str, Any]:
    payload = json.loads(scene_json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("scene_xml"), str):
        return payload
    return payload if isinstance(payload, dict) else {}


def resolve_scene_xml(scene_path: Path) -> Path:
    if scene_path.is_file() and scene_path.suffix.lower() == ".xml":
        return scene_path
    if scene_path.is_file() and scene_path.suffix.lower() == ".json":
        payload = parse_scene_json(scene_path)
        scene_xml_text = str(payload.get("scene_xml", "")).strip()
        if not scene_xml_text:
            raise ValueError(f"Scene JSON does not contain scene_xml: {scene_path}")
        scene_xml = Path(scene_xml_text).expanduser()
        if not scene_xml.is_absolute():
            scene_xml = PROJECT_ROOT / scene_xml
        return scene_xml
    raise ValueError(f"Unsupported scene path: {scene_path}")


def load_qpos_from_log_dir(log_dir: Path) -> np.ndarray | None:
    states_path = log_dir / "states.npy.zst"
    info_path = log_dir / "info.json"
    if not states_path.exists() or not info_path.exists():
        return None
    with open(states_path, "rb") as f:
        with zstd.ZstdDecompressor().stream_reader(f) as zstd_f:
            states_io = BytesIO(zstd_f.read())
    states = np.load(states_io)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    split = info.get("split", {}).get("qpos")
    if not isinstance(split, dict):
        return None
    start = int(split["start"])
    end = int(split["end"])
    shape = tuple(split["shape"])
    qpos = states[..., start:end].reshape(states.shape[:-1] + shape)
    return qpos.astype(np.float64)


def load_scene_source(source: Path) -> tuple[mujoco.MjModel, np.ndarray | None, Path]:
    source = Path(source)
    if source.is_dir():
        model_path = source / "model.mjb"
        if not model_path.exists():
            model_path = source / ".." / "model.mjb"
        if model_path.exists():
            model = mujoco.MjModel.from_binary_path(str(model_path))
        else:
            scene_xml = source / "scene.xml"
            if not scene_xml.exists():
                raise FileNotFoundError(f"Could not find model.mjb or scene.xml under {source}")
            model = mujoco.MjModel.from_xml_path(str(scene_xml))
        return model, load_qpos_from_log_dir(source), source

    if source.suffix.lower() == ".json":
        scene_xml = resolve_scene_xml(source)
        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        return model, None, source.parent

    if source.suffix.lower() == ".xml":
        model = mujoco.MjModel.from_xml_path(str(source))
        qpos = load_qpos_from_log_dir(source.parent)
        return model, qpos, source.parent

    raise ValueError(f"Unsupported source: {source}")


def load_qpos_npy(qpos_path: Path) -> np.ndarray:
    qpos = np.load(qpos_path)
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    return qpos


class MeshManager:
    def __init__(self, model: mujoco.MjModel, mesh_dir: Path, scratch: bpy.types.Collection, primitive_resolution: int = 16):
        self.model = model
        self.mesh_dir = mesh_dir
        self.scratch = scratch
        self.primitive_resolution = primitive_resolution
        self.blender_plane = None
        self.blender_sphere = None
        self.blender_capsule = {}
        self.blender_cylinder = None
        self.blender_box = None
        self.blender_meshes = {}

    def add_plane(self, half_width: float, half_height: float):
        if self.blender_plane is None:
            vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
            faces = np.array([[0, 1, 2, 3]], dtype=int)
            self.blender_plane = add_mesh("primitive.plane", vertices, faces)
        return self.blender_plane, np.array((half_width, half_height, 1.0), dtype=float)

    def add_sphere(self, radius: float):
        if self.blender_sphere is None:
            mesh = trimesh.creation.uv_sphere(count=[self.primitive_resolution, self.primitive_resolution])
            self.blender_sphere = add_mesh("primitive.sphere", mesh.vertices, mesh.faces)
        return self.blender_sphere, np.array((radius, radius, radius), dtype=float)

    def add_capsule(self, radius: float, half_height: float):
        ratio = half_height / radius
        if ratio not in self.blender_capsule:
            mesh = trimesh.creation.capsule(radius=1, height=2 * ratio, count=[self.primitive_resolution, self.primitive_resolution])
            self.blender_capsule[ratio] = add_mesh("primitive.capsule", mesh.vertices, mesh.faces)
        return self.blender_capsule[ratio], np.array((radius, radius, radius), dtype=float)

    def add_ellipsoid(self, radii: np.ndarray):
        if self.blender_sphere is None:
            mesh = trimesh.creation.uv_sphere(count=[self.primitive_resolution, self.primitive_resolution])
            self.blender_sphere = add_mesh("primitive.sphere", mesh.vertices, mesh.faces)
        return self.blender_sphere, radii

    def add_cylinder(self, radius: float, half_height: float):
        if self.blender_cylinder is None:
            mesh = trimesh.creation.cylinder(radius=1, height=2, count=[self.primitive_resolution, self.primitive_resolution])
            self.blender_cylinder = add_mesh("primitive.cylinder", mesh.vertices, mesh.faces)
        return self.blender_cylinder, np.array((radius, radius, half_height), dtype=float)

    def add_box(self, half_extents: np.ndarray):
        if self.blender_box is None:
            mesh = trimesh.creation.box(extents=(2,) * 3)
            self.blender_box = add_mesh("primitive.box", mesh.vertices, mesh.faces)
        return self.blender_box, half_extents

    def add_mesh(self, mesh_id: int, prefix: str = ""):
        if mesh_id in self.blender_meshes:
            return self.blender_meshes[mesh_id]
        mesh = self.model.mesh(mesh_id)
        name = remove_prefix(mesh.name, prefix) if prefix and mesh.name.startswith(prefix) else mesh.name
        try:
            ret = self._add_mesh_from_file(mesh_id, name)
        except FileNotFoundError:
            ret = self._add_mesh_from_data(mesh_id, mesh.name)
        self.blender_meshes[mesh_id] = ret
        return ret

    def _add_mesh_from_file(self, mesh_id: int, name: str):
        pathadr = self.model.mesh_pathadr[mesh_id].item()
        if pathadr == -1:
            raise FileNotFoundError
        paths = self.model.paths[pathadr:]
        null = paths.index(0)
        path = self.mesh_dir / paths[:null].decode("utf-8")

        assert len(self.scratch.objects) == 0
        with temporary_active_collection(self.scratch):
            import_mesh(path)

        meshes = [obj.data for obj in self.scratch.objects if obj.type == "MESH"]
        if len(meshes) != 1:
            raise ValueError(f"Expected 1 mesh, found {len(meshes)} for {path}")

        mesh = meshes[0]
        mesh.name = name
        mesh.materials.append(None)

        pos = np.array(self.model.mesh_pos[mesh_id], dtype=float)
        quat = np.array(self.model.mesh_quat[mesh_id], dtype=float)
        scale = np.array(self.model.mesh_scale[mesh_id], dtype=float)

        for obj in list(self.scratch.objects):
            bpy.data.objects.remove(obj, do_unlink=True)

        return mesh, pos, quat, scale

    def _add_mesh_from_data(self, mesh_id: int, name: str):
        mesh = self.model.mesh(mesh_id)
        vertadr = mesh.vertadr.item()
        vertnum = mesh.vertnum.item()
        faceadr = mesh.faceadr.item()
        facenum = mesh.facenum.item()
        vertices = np.asarray(self.model.mesh_vert[vertadr : vertadr + vertnum], dtype=float)
        faces = np.asarray(self.model.mesh_face[faceadr : faceadr + facenum], dtype=int)
        blender_mesh = add_mesh(name, vertices, faces)
        return blender_mesh, np.array((0.0, 0.0, 0.0), dtype=float), np.array((1.0, 0.0, 0.0, 0.0), dtype=float), np.array((1.0, 1.0, 1.0), dtype=float)


def add_geometry(
    model: mujoco.MjModel,
    blender_bodies: dict[int, bpy.types.Object],
    target_collection: bpy.types.Collection,
    mesh_dir: Path,
    body_prefix: dict[int, str] | None = None,
):
    scratch = ensure_collection("Scratch")
    missing_material = ensure_material("Missing", np.array([1.0, 0.0, 1.0, 1.0], dtype=float))
    if body_prefix is None:
        body_prefix = {i: "" for i in blender_bodies}

    mesh_manager = MeshManager(model, mesh_dir, scratch)
    materials: dict[int, bpy.types.Material] = {}

    for i in range(model.ngeom):
        geom = model.geom(i)
        bodyid = geom.bodyid.item()
        if bodyid not in blender_bodies or geom.group >= 3:
            continue

        blender_body = blender_bodies[bodyid]
        pos = np.array(geom.pos, dtype=float)
        quat = np.array(geom.quat, dtype=float)
        geom_type = int(np.asarray(geom.type).item())

        if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            half_width, half_height, _ = geom.size
            mesh, scale = mesh_manager.add_plane(half_width, half_height)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_HFIELD):
            raise NotImplementedError("HFIELD geoms are not supported")
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            radius, _, _ = geom.size
            mesh, scale = mesh_manager.add_sphere(radius)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            radius, half_height, _ = geom.size
            mesh, scale = mesh_manager.add_capsule(radius, half_height)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            radii = np.array(geom.size, dtype=float)
            mesh, scale = mesh_manager.add_ellipsoid(radii)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            radius, half_height, _ = geom.size
            mesh, scale = mesh_manager.add_cylinder(radius, half_height)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            half_extents = np.array(geom.size, dtype=float)
            mesh, scale = mesh_manager.add_box(half_extents)
        elif geom_type in (int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_SDF)):
            dataid = geom.dataid.item()
            mesh, mesh_pos, mesh_quat, scale = mesh_manager.add_mesh(dataid, body_prefix[bodyid])
            pos, quat = compute_parent_transform(pos, quat, mesh_pos, mesh_quat)
        else:
            raise NotImplementedError(f"Unsupported geom type: {geom_type}")

        matid = geom.matid.item()
        if matid == -1:
            blender_material = missing_material
        elif matid in materials:
            blender_material = materials[matid]
        else:
            material = model.mat(matid)
            material_name = remove_prefix(material.name, body_prefix[bodyid]) if material.name.startswith(body_prefix[bodyid]) else material.name
            blender_material = ensure_material(material_name, np.array(material.rgba, dtype=float))
            materials[matid] = blender_material

        if geom.name == "":
            name = mesh.name
        else:
            name = remove_prefix(geom.name, body_prefix[bodyid]) if geom.name.startswith(body_prefix[bodyid]) else geom.name
        add_object(
            name=name,
            data=mesh,
            parent=blender_body,
            collection=target_collection,
            location=pos,
            rotation_quaternion=quat,
            scale=scale,
            material=blender_material,
            allow_rename=True,
        )


def add_hierarchy(
    model: mujoco.MjModel,
    mesh_dir: Path,
    camera_name: str | None = None,
):
    workspace = bpy.data.collections.new("Workspace")
    bpy.context.scene.collection.children.link(workspace)

    blender_bodies: dict[int, bpy.types.Object] = {}
    body_copied: dict[int, bool] = {}
    cameras: dict[str, bpy.types.Object] = {}

    for i in range(1, model.nbody):
        body = model.body(i)
        parentid = body.parentid.item()
        blender_parent = None if parentid == 0 else blender_bodies[parentid]

        blender_body = add_object(
            name=sanitize_key(body.name),
            parent=blender_parent,
            collection=workspace,
            location=np.array(body.pos, dtype=float),
            rotation_quaternion=np.array(body.quat, dtype=float),
            allow_rename=True,
        )
        blender_body.empty_display_size = 0.1
        blender_bodies[i] = blender_body
        body_copied[i] = False

    add_geometry(model, blender_bodies, workspace, mesh_dir)

    for i in range(model.ncam):
        camera = model.cam(i)
        if camera.mode != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
            continue
        fovy = np.deg2rad(float(camera.fovy.item()))
        clip_start = float(model.vis.map.znear * model.stat.extent)
        clip_end = float(model.vis.map.zfar * model.stat.extent)
        camera_data = add_fixed_camera(name=sanitize_key(camera.name), fovy=fovy, clip_start=clip_start, clip_end=clip_end)
        bodyid = camera.bodyid.item()
        camera_body = blender_bodies.get(bodyid)
        blender_camera = add_object(
            name=sanitize_key(camera.name),
            data=camera_data,
            collection=workspace,
            parent=camera_body,
            location=np.array(camera.pos, dtype=float),
            rotation_quaternion=np.array(camera.quat, dtype=float),
            allow_rename=True,
        )
        cameras[str(camera.name)] = blender_camera

    if camera_name is None and cameras:
        camera_name = next(iter(cameras.keys()))
    if camera_name is None or camera_name not in cameras:
        raise RuntimeError("No fixed camera found in scene")
    bpy.context.scene.camera = cameras[camera_name]
    return blender_bodies, cameras


def add_animation(model: mujoco.MjModel, qpos: np.ndarray, blender_bodies: dict[int, bpy.types.Object], fps: int):
    assert qpos.ndim == 2 and qpos.shape[1] == model.nq, f"Invalid qpos shape: {qpos.shape}"
    timestep = model.opt.timestep
    num_steps = qpos.shape[0]
    step = 1 / fps / timestep
    if not np.isclose(step, int(step)):
        print(f"Warning: Inexact step size {step} for timestep {timestep} and fps {fps}")
    indices = np.arange(0, num_steps, step)
    indices = np.rint(indices).astype(int)
    qpos = qpos[indices]
    num_frames = len(indices)
    frames = np.arange(num_frames, dtype=np.float64)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames - 1
    bpy.context.scene.render.fps = fps
    bpy.context.scene.frame_set(0)

    data = mujoco.MjData(model)
    xpos = np.empty((num_frames, model.nbody, 3))
    xquat = np.empty((num_frames, model.nbody, 4))
    for i in range(num_frames):
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        xpos[i] = data.xpos
        xquat[i] = data.xquat

    action = bpy.data.actions.new(name="kinematics")
    action_layer = action.layers.new(name="kinematics")
    action_strip = action_layer.strips.new(type="KEYFRAME")
    for i in blender_bodies:
        body = model.body(i)
        if body.weldid != i:
            continue
        parentid = body.parentid.item()
        relative_pos, relative_quat = compute_relative_transform(
            xpos[:, i], xquat[:, i],
            xpos[:, parentid], xquat[:, parentid],
        )

        action_slot = action.slots.new("OBJECT", body.name)
        channelbag = action_strip.channelbags.new(action_slot)
        fcurves = channelbag.fcurves
        jntadr = body.jntadr.item()
        jntnum = body.jntnum.item()
        translation = False
        rotation = False
        for j in range(jntadr, jntadr + jntnum):
            joint = model.joint(j)
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                translation = rotation = True
            elif joint.type == mujoco.mjtJoint.mjJNT_BALL:
                rotation = True
                translation = not np.all(joint.pos == 0)
            elif joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                translation = True
            elif joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                rotation = True
                translation = not np.all(joint.pos == 0)
        if translation:
            for j in range(3):
                fcurve = fcurves.new(data_path="location", index=j)
                fcurve.keyframe_points.add(num_frames)
                co = np.stack((frames, relative_pos[:, j]), axis=-1)
                fcurve.keyframe_points.foreach_set("co", co.ravel())
        if rotation:
            for j in range(4):
                fcurve = fcurves.new(data_path="rotation_quaternion", index=j)
                fcurve.keyframe_points.add(num_frames)
                co = np.stack((frames, relative_quat[:, j]), axis=-1)
                fcurve.keyframe_points.foreach_set("co", co.ravel())

        blender_body = blender_bodies[i]
        blender_body.animation_data_create()
        blender_body.animation_data.action = action


def render_frame_sequence(
    cameras: dict[str, bpy.types.Object],
    frame_count: int,
    frames_dir: Path,
    wrist_frames_dir: Path | None = None,
    main_camera_name: str = "table_cam_front",
    wrist_camera_name: str = "/ur:wrist_cam",
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    if wrist_frames_dir is not None:
        wrist_frames_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    main_camera = cameras.get(main_camera_name) or next(iter(cameras.values()), None)
    if main_camera is None:
        raise RuntimeError("No Blender camera available for rendering")
    wrist_camera = cameras.get(wrist_camera_name, main_camera)

    for frame_idx in range(frame_count):
        scene.frame_set(frame_idx)

        scene.camera = main_camera
        scene.render.filepath = str(frames_dir / f"{frame_idx:06d}.png")
        bpy.ops.render.render(write_still=True)

        if wrist_frames_dir is not None:
            scene.camera = wrist_camera
            scene.render.filepath = str(wrist_frames_dir / f"{frame_idx:06d}.png")
            bpy.ops.render.render(write_still=True)


def set_keyframe(model: mujoco.MjModel, qpos: np.ndarray, blender_bodies: dict[int, bpy.types.Object]):
    assert qpos.shape == (model.nq,), f"Invalid qpos shape: {qpos.shape}"
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    xpos = data.xpos
    xquat = data.xquat
    for i in blender_bodies:
        body = model.body(i)
        if body.weldid != i:
            continue
        parentid = body.parentid.item()
        relative_pos, relative_quat = compute_relative_transform(
            xpos[i], xquat[i],
            xpos[parentid], xquat[parentid],
        )
        blender_body = blender_bodies[i]
        blender_body.location = relative_pos
        blender_body.rotation_mode = "QUATERNION"
        blender_body.rotation_quaternion = relative_quat


def ensure_default_lighting() -> None:
    scene = bpy.context.scene
    has_light = any(obj.type == "LIGHT" for obj in bpy.data.objects)
    if not has_light:
        light_data = bpy.data.lights.new(name="AutoSun", type="SUN")
        light_data.energy = 3.0
        light_obj = bpy.data.objects.new(name="AutoSun", object_data=light_data)
        scene.collection.objects.link(light_obj)
        light_obj.location = (2.0, -2.0, 4.0)
        light_obj.rotation_euler = (0.9, 0.0, 0.8)

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Strength"].default_value = max(0.8, float(bg.inputs["Strength"].default_value))


def configure_render_settings(output_path: Path, fps: int, width: int, height: int) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.fps = fps
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except Exception:
        try:
            scene.render.engine = "CYCLES"
        except Exception:
            pass

    if output_path.suffix.lower() in {".mp4", ".mkv", ".mov"}:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.filepath = str(output_path)
    else:
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(output_path)


WORKER_PROTOCOL_PREFIX = "<<RWORKER>>"


def _apply_worker_render_speed_tunings(scene) -> None:
    """Apply lightweight tunings so per-frame worker renders stay fast."""
    try:
        scene.render.use_simplify = True
        scene.render.simplify_subdivision = 0
        scene.render.simplify_child_particles = 0.0
        scene.render.resolution_percentage = 100
    except Exception:
        pass
    if hasattr(scene, "cycles"):
        try:
            scene.cycles.samples = 16
        except Exception:
            pass
        try:
            scene.cycles.use_denoising = False
        except Exception:
            pass
        try:
            scene.cycles.max_bounces = 2
        except Exception:
            pass
    if hasattr(scene, "eevee"):
        for attr, value in (
            ("taa_render_samples", 8),
            ("use_gtao", False),
            ("use_bloom", False),
            ("use_ssr", False),
        ):
            try:
                setattr(scene.eevee, attr, value)
            except Exception:
                continue


def run_worker(
    source: Path,
    *,
    width: int,
    height: int,
    main_camera_name: str,
    wrist_camera_name: str,
) -> int:
    """Persistent Blender worker.

    Loads the scene once, then reads ``<<RWORKER>> {json}`` commands from stdin
    and writes ``<<RWORKER>> OK|ERR ...`` markers to stdout. The parent process
    drives this worker to render single camera frames on demand without paying
    Blender's startup cost per frame.
    """
    import sys as _sys

    source = source.expanduser()
    if not source.is_absolute():
        source = PROJECT_ROOT / source

    width = int(max(64, width))
    height = int(max(64, height))

    model, _, _ = load_scene_source(source)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    ensure_default_lighting()
    blender_bodies, cameras = add_hierarchy(model, ASSET_ROOT, camera_name=main_camera_name)

    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.image_settings.file_format = "PNG"
    requested_engine = (os.environ.get("BLENDER_RENDER_ENGINE", "BLENDER_EEVEE_NEXT") or "BLENDER_EEVEE_NEXT").upper()
    try:
        scene.render.engine = requested_engine
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            try:
                scene.render.engine = "CYCLES"
            except Exception:
                pass
    _apply_worker_render_speed_tunings(scene)

    main_camera = cameras.get(main_camera_name) or next(iter(cameras.values()), None)
    wrist_camera = cameras.get(wrist_camera_name)

    print(f"{WORKER_PROTOCOL_PREFIX} READY", flush=True)

    for raw_line in _sys.stdin:
        line = raw_line.rstrip()
        if not line.startswith(WORKER_PROTOCOL_PREFIX):
            continue
        payload_text = line[len(WORKER_PROTOCOL_PREFIX) :].strip()
        if not payload_text:
            continue
        try:
            request = json.loads(payload_text)
        except Exception as exc:
            print(f"{WORKER_PROTOCOL_PREFIX} ERR json:{exc}", flush=True)
            continue
        cmd = str(request.get("cmd", "")).strip()
        if cmd == "exit":
            print(f"{WORKER_PROTOCOL_PREFIX} BYE", flush=True)
            return 0
        if cmd != "render":
            print(f"{WORKER_PROTOCOL_PREFIX} ERR unknown_cmd:{cmd}", flush=True)
            continue
        try:
            qpos = np.asarray(request.get("qpos", []), dtype=np.float64)
            if qpos.shape != (model.nq,):
                raise ValueError(f"qpos shape {tuple(qpos.shape)} != ({model.nq},)")
            set_keyframe(model, qpos, blender_bodies)
            front_path = request.get("front") or ""
            wrist_path = request.get("wrist") or ""
            if front_path:
                if main_camera is None:
                    raise RuntimeError("Main camera not available in scene")
                scene.camera = main_camera
                scene.render.filepath = str(front_path)
                bpy.ops.render.render(write_still=True)
            if wrist_path:
                target_camera = wrist_camera or main_camera
                if target_camera is None:
                    raise RuntimeError("No camera available for wrist render")
                scene.camera = target_camera
                scene.render.filepath = str(wrist_path)
                bpy.ops.render.render(write_still=True)
            print(f"{WORKER_PROTOCOL_PREFIX} OK", flush=True)
        except Exception as exc:
            print(f"{WORKER_PROTOCOL_PREFIX} ERR render:{exc}", flush=True)
    return 0


def render_source(
    source: Path,
    output: Path,
    *,
    fps: int = 20,
    width: int = 1280,
    height: int = 960,
    camera_name: str | None = None,
    keyframe: int = -1,
    save_blend: Path | None = None,
    qpos_path: Path | None = None,
    frames_dir: Path | None = None,
    wrist_frames_dir: Path | None = None,
    blend_only: bool = False,
) -> Path:
    model, qpos_from_source, _ = load_scene_source(source)
    qpos = load_qpos_npy(qpos_path) if qpos_path is not None else qpos_from_source

    bpy.ops.wm.read_factory_settings(use_empty=True)
    ensure_default_lighting()
    blender_bodies, cameras = add_hierarchy(model, ASSET_ROOT, camera_name=camera_name)

    if qpos is not None and qpos.ndim == 2 and keyframe < 0:
        add_animation(model, qpos, blender_bodies, fps=fps)
    elif qpos is not None and qpos.ndim == 2 and keyframe >= 0:
        idx = int(np.clip(keyframe, 0, qpos.shape[0] - 1))
        set_keyframe(model, qpos[idx], blender_bodies)

    if save_blend is None:
        save_blend = (source if source.is_dir() else output.parent) / "scene.blend"
    save_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(save_blend), check_existing=False, compress=True)
    if blend_only:
        return save_blend

    configure_render_settings(output, fps=fps, width=width, height=height)
    if qpos is not None and qpos.ndim == 2 and keyframe < 0 and frames_dir is not None:
        render_frame_sequence(
            cameras=cameras,
            frame_count=qpos.shape[0],
            frames_dir=frames_dir,
            wrist_frames_dir=wrist_frames_dir,
            main_camera_name=str(camera_name or "table_cam_front"),
        )
    else:
        scene = bpy.context.scene
        scene.frame_set(max(0, int(keyframe)))
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
    return output


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Render MuJoCo scenes/logs with Blender")
    parser.add_argument("source_positional", nargs="?", type=Path, help="Scene XML/JSON or rollout log directory")
    parser.add_argument("--source", type=Path, default=None, help="Scene XML/JSON or rollout log directory")
    parser.add_argument("--output", type=Path, default=Path("logs/blender_render.mp4"), help="Output video or image path")
    parser.add_argument("--blend-out", type=Path, default=None, help="Optional .blend output path")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--camera", type=str, default="table_cam_front")
    parser.add_argument("--wrist-camera", type=str, default="/ur:wrist_cam")
    parser.add_argument("--keyframe", type=int, default=-1, help="Render a single keyframe from qpos instead of animation")
    parser.add_argument("--qpos-npy", type=Path, default=None, help="Optional qpos trajectory .npy for rendering frames")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Optional output directory for front frames")
    parser.add_argument("--wrist-frames-dir", type=Path, default=None, help="Optional output directory for wrist frames")
    parser.add_argument("--blend-only", action="store_true", help="Only build hierarchy/animation and write .blend")
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run as a persistent stdin/stdout render worker for live per-frame rendering",
    )
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    source_arg = args.source if args.source is not None else args.source_positional
    if source_arg is None:
        raise ValueError("Missing source path. Provide either positional source or --source.")
    source = source_arg.expanduser()
    if not source.is_absolute():
        source = PROJECT_ROOT / source

    if args.worker:
        rc = run_worker(
            source,
            width=int(args.width),
            height=int(args.height),
            main_camera_name=str(args.camera or "table_cam_front"),
            wrist_camera_name=str(args.wrist_camera or "/ur:wrist_cam"),
        )
        sys.exit(int(rc))
    output = args.output.expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    blend_out = args.blend_out.expanduser() if args.blend_out is not None else None
    if blend_out is not None and not blend_out.is_absolute():
        blend_out = PROJECT_ROOT / blend_out
    if blend_out is None:
        blend_out = (source if source.is_dir() else output.parent) / "scene.blend"
    qpos_npy = args.qpos_npy
    if qpos_npy is None and source.is_dir():
        candidate_qpos = source / "qpos.npy"
        if candidate_qpos.exists():
            qpos_npy = candidate_qpos
    if qpos_npy is not None:
        qpos_npy = qpos_npy.expanduser()
        if not qpos_npy.is_absolute():
            qpos_npy = PROJECT_ROOT / qpos_npy

    render_source(
        source,
        output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_name=args.camera,
        keyframe=args.keyframe,
        save_blend=blend_out,
        qpos_path=qpos_npy,
        frames_dir=args.frames_dir,
        wrist_frames_dir=args.wrist_frames_dir,
        blend_only=bool(args.blend_only),
    )


if __name__ == "__main__":
    main()