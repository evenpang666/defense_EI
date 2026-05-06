import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import imageio.v2 as imageio
import mujoco
import numpy as np

def _try_load_optional_mjlab_plugin() -> None:
    loader = getattr(mujoco, "mj_loadPluginLibrary", None)
    if not callable(loader):
        return
    candidates: list[str] = []
    env_candidate = str(os.environ.get("MJLAB_PLUGIN_PATH", "")).strip()
    if env_candidate:
        candidates.append(env_candidate)
    candidates.extend(["./libmjlab.so.3.3.0", "./mjlab.dll", "./libmjlab.dylib"])
    for plugin_path in candidates:
        try:
            loader(plugin_path)
            return
        except Exception:
            continue


_try_load_optional_mjlab_plugin()


# =========================
# 路径设置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "model"
LOG_ROOT = PROJECT_ROOT / "logs"
VIEW_LOG_DIR = LOG_ROOT / "simple_views"

VIEW_IMAGE_WIDTH = 960
VIEW_IMAGE_HEIGHT = 1280
DEFAULT_SCENE_JSON = LOG_ROOT / "exported_scene.json"


# =========================
# 工具函数
# =========================
def parse_root_body_name(xml_path: Path) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing <worldbody> in {xml_path}")
    bodies = worldbody.findall("body")
    if not bodies:
        raise ValueError(f"No <body> found in {xml_path}")
    names = [b.get("name") for b in bodies if b.get("name")]
    if "world" in names:
        return "world"
    return names[0]


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


def resolve_input_path(file_obj, path_text: str | None) -> Path | None:
    if file_obj is not None:
        candidate = getattr(file_obj, "name", None) or str(file_obj)
        if candidate:
            return Path(candidate)

    text = (path_text or "").strip()
    if not text:
        return None

    p = Path(text)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def relpath_posix(target: Path, base: Path) -> str:
    try:
        return Path(target).resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        import os
        return Path(os.path.relpath(str(target.resolve()), str(base.resolve()))).as_posix()


def scale_numbers(values: list[float], scale: np.ndarray) -> list[float]:
    if not values:
        return values
    out = []
    for i, v in enumerate(values):
        out.append(float(v) * float(scale[min(i, 2)]))
    return out


def scale_attr(elem: ET.Element, attr: str, scale: np.ndarray):
    if attr not in elem.attrib:
        return
    vals = parse_float_list(elem.get(attr), [])
    if not vals:
        return
    elem.set(attr, format_float_list(scale_numbers(vals, scale)))


def scale_geom_size(elem: ET.Element, scale: np.ndarray):
    if "size" not in elem.attrib:
        return
    vals = parse_float_list(elem.get("size"), [])
    if not vals:
        return

    gtype = (elem.get("type") or "").lower()
    if gtype == "sphere" and len(vals) == 1:
        uniform = float(np.mean(scale))
        elem.set("size", f"{vals[0] * uniform:.6f}")
        return
    if gtype in {"capsule", "cylinder"} and len(vals) == 2:
        radius_scale = float((scale[0] + scale[1]) / 2.0)
        elem.set("size", format_float_list([vals[0] * radius_scale, vals[1] * scale[2]]))
        return

    elem.set("size", format_float_list(scale_numbers(vals, scale)))


def transform_asset_tree_for_scale(root: ET.Element, scale: np.ndarray):
    for elem in root.iter():
        tag = elem.tag
        if tag in {"body", "geom", "site", "camera", "light", "inertial", "joint"}:
            scale_attr(elem, "pos", scale)

        if tag == "geom":
            scale_geom_size(elem, scale)
            scale_attr(elem, "fromto", scale)

        elif tag == "site":
            scale_attr(elem, "size", scale)
            scale_attr(elem, "fromto", scale)

        elif tag == "mesh":
            current = parse_float_list(elem.get("scale"), [1.0, 1.0, 1.0])
            if len(current) == 1:
                current = current * 3
            if len(current) == 2:
                current = [current[0], current[1], 1.0]
            current = np.asarray(current[:3], dtype=float)
            new_scale = current * scale
            elem.set("scale", format_float_list(new_scale))


# =========================
# 数据结构
# =========================
@dataclass
class AssetDef:
    key: str
    xml_path: Path
    root_body_name: str


@dataclass
class PlacedAsset:
    id: int
    key: str
    model_name: str
    joint_name: str
    prefix: str
    pos: np.ndarray
    quat: np.ndarray
    scale: np.ndarray


# =========================
# 简化运行时
# =========================
class SimpleRuntime:
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="evobody_simple_"))
        self.runtime_xml = self.temp_dir / "simple_runtime.xml"
        self.generated_runtime_assets_dir = self.temp_dir / "runtime_assets"

        self.assets = self._discover_assets()
        self.safe_key_to_key = {sanitize_key(k): k for k in self.assets.keys()}

        self.placed_assets: list[PlacedAsset] = []
        self._next_asset_id = 1

        self.robot_base_pos = np.array([0.0, 0.0, 0.824], dtype=float)
        self.robot_base_quat = np.array([0.0, 0.0, 0.0, -1.0], dtype=float)

        self.camera_pos = np.array([0.0, -1.4, 1.45], dtype=float)
        self.camera_quat = np.array([0.819, 0.574, 0.0, 0.0], dtype=float)

        self.model = None
        self.data = None

        self._write_runtime_xml()
        self._load_model()

    def _discover_assets(self) -> dict[str, AssetDef]:
        results: dict[str, AssetDef] = {}
        candidates = []
        candidates.extend((MODEL_ROOT / "object").glob("*.xml"))
        candidates.extend((MODEL_ROOT / "instrument").glob("*.xml"))

        for path in sorted(candidates):
            try:
                root_body = parse_root_body_name(path)
            except Exception:
                continue
            key = f"{path.parent.name}/{path.stem}"
            results[key] = AssetDef(key=key, xml_path=path, root_body_name=root_body)

        return results

    def _append_loaded_asset(self, key: str, pos: np.ndarray, quat: np.ndarray, scale: np.ndarray):
        asset_id = self._next_asset_id
        self._next_asset_id += 1
        self.placed_assets.append(
            PlacedAsset(
                id=asset_id,
                key=key,
                model_name=f"user_model_{asset_id}",
                joint_name=f"asset_{asset_id}_joint",
                prefix=f"user{asset_id}/",
                pos=pos,
                quat=quat,
                scale=scale,
            )
        )

    def _instance_asset_xml_path(self, item: PlacedAsset) -> Path:
        asset_def = self.assets[item.key]
        sx, sy, sz = [float(x) for x in item.scale.tolist()]
        name = f"{asset_def.xml_path.stem}__scaled__id{item.id}__s_{sx:.4f}_{sy:.4f}_{sz:.4f}.xml"
        return self.generated_runtime_assets_dir / name

    def _write_scaled_asset_xml(self, item: PlacedAsset) -> Path:
        asset_def = self.assets[item.key]

        if np.allclose(item.scale, np.ones(3, dtype=float)):
            return asset_def.xml_path

        out_path = self._instance_asset_xml_path(item)
        tree = ET.parse(asset_def.xml_path)
        root = tree.getroot()
        transform_asset_tree_for_scale(root, item.scale)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_path, encoding="utf-8", xml_declaration=False)
        return out_path

    def load_scene_json(self, json_path: Path):
        if not json_path.exists():
            raise FileNotFoundError(f"JSON scene not found: {json_path}")

        content = json.loads(json_path.read_text(encoding="utf-8"))
        assets = content.get("assets", [])
        robot = content.get("robot", {})
        camera = content.get("camera", {})

        self.robot_base_pos = np.asarray(robot.get("base_pos", [0.0, 0.0, 0.824]), dtype=float)
        self.robot_base_quat = np.asarray(robot.get("base_quat", [0.0, 0.0, 0.0, -1.0]), dtype=float)

        self.camera_pos = np.asarray(camera.get("pos", [0.0, -1.4, 1.45]), dtype=float)
        self.camera_quat = np.asarray(camera.get("quat", [0.819, 0.574, 0.0, 0.0]), dtype=float)

        if self.robot_base_pos.shape != (3,):
            self.robot_base_pos = np.array([0.0, 0.0, 0.824], dtype=float)
        if self.robot_base_quat.shape != (4,) or np.linalg.norm(self.robot_base_quat) < 1e-8:
            self.robot_base_quat = np.array([0.0, 0.0, 0.0, -1.0], dtype=float)

        if self.camera_pos.shape != (3,):
            self.camera_pos = np.array([0.0, -1.4, 1.45], dtype=float)
        if self.camera_quat.shape != (4,) or np.linalg.norm(self.camera_quat) < 1e-8:
            self.camera_quat = np.array([0.819, 0.574, 0.0, 0.0], dtype=float)

        self.robot_base_quat = self.robot_base_quat / np.linalg.norm(self.robot_base_quat)
        self.camera_quat = self.camera_quat / np.linalg.norm(self.camera_quat)

        self.placed_assets.clear()
        self._next_asset_id = 1

        for entry in assets:
            key = entry.get("key")
            if key not in self.assets:
                print(f"[Scene] Skip unknown asset: {key}")
                continue

            pos = np.asarray(entry.get("pos", [0.0, 0.0, 0.845]), dtype=float)
            quat = np.asarray(entry.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float)
            scale = np.asarray(entry.get("scale", [1.0, 1.0, 1.0]), dtype=float)

            if pos.shape != (3,):
                pos = np.array([0.0, 0.0, 0.845], dtype=float)
            if quat.shape != (4,) or np.linalg.norm(quat) < 1e-8:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if scale.shape != (3,):
                scale = np.array([1.0, 1.0, 1.0], dtype=float)

            quat = quat / np.linalg.norm(quat)
            scale = np.clip(scale.astype(float), 0.05, 50.0)

            self._append_loaded_asset(key, pos, quat, scale)

        self._write_runtime_xml()
        self._load_model()

    def _build_scene_xml_text(self, base_dir: Path) -> str:
        base_pos = format_float_list(self.robot_base_pos)
        base_quat = format_float_list(self.robot_base_quat)
        cam_pos = format_float_list(self.camera_pos)
        cam_quat = format_float_list(self.camera_quat)

        lines = [
            '<mujoco model="simple_viewer">',
            '  <option integrator="implicitfast" impratio="10" cone="elliptic" noslip_iterations="2">',
            '    <flag multiccd="enable"/>',
            '  </option>',
            '  <visual>',
            '    <global azimuth="220" elevation="-30" offwidth="1280" offheight="960"/>',
            '  </visual>',
            '  <asset>',
            '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>',
            '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.6 0.7 0.8" rgb2="0.4 0.5 0.6" markrgb="0.8 0.8 0.8" width="300" height="300"/>',
            '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>',
            f'    <model name="desk_model" file="{relpath_posix(MODEL_ROOT / "misc" / "simple_table.xml", base_dir)}" content_type="text/xml"/>',
            f'    <model name="ur5e_model" file="{relpath_posix(MODEL_ROOT / "robot" / "ur5e_gripper.xml", base_dir)}" content_type="text/xml"/>',
        ]

        for item in self.placed_assets:
            scaled_xml = self._write_scaled_asset_xml(item)
            lines.append(
                f'    <model name="{item.model_name}" file="{relpath_posix(scaled_xml, base_dir)}" content_type="text/xml"/>'
            )

        lines.extend([
            '  </asset>',
            '  <worldbody>',
            '    <light directional="true" diffuse="0.8 0.8 0.8" ambient="0.2 0.2 0.2" pos="0 0 5" dir="0 0 -1"/>',
            '    <geom name="floor" pos="0 0 0" size="2.5 2.5 0.05" type="plane" material="groundplane"/>',
            '    <body name="desk" pos="0 0 0" quat="1 0 0 1">',
            '      <attach model="desk_model" body="vention table" prefix="desk/"/>',
            f'      <camera name="table_cam_front" pos="{cam_pos}" quat="{cam_quat}" fovy="45" resolution="1280 960"/>',
            '    </body>',
            f'    <body name="ur5e_center" pos="{base_pos}" quat="{base_quat}">',
            '      <attach model="ur5e_model" body="world" prefix="/ur:"/>',
            '    </body>',
        ])

        for item in self.placed_assets:
            asset = self.assets[item.key]
            pos = format_float_list(item.pos)
            quat = format_float_list(item.quat)
            lines.extend([
                f'    <body name="asset_{item.id}" pos="{pos}" quat="{quat}">',
                f'      <joint name="{item.joint_name}" type="free"/>',
                f'      <attach model="{item.model_name}" body="{asset.root_body_name}" prefix="{item.prefix}"/>',
                '    </body>',
            ])

        lines.extend([
            '  </worldbody>',
            '</mujoco>',
        ])

        return "\n".join(lines)

    def _write_runtime_xml(self):
        self.generated_runtime_assets_dir.mkdir(parents=True, exist_ok=True)
        xml_text = self._build_scene_xml_text(self.temp_dir)
        self.runtime_xml.write_text(xml_text, encoding="utf-8")

    def _load_model(self):
        self.model = mujoco.MjModel.from_xml_path(str(self.runtime_xml))
        self.data = mujoco.MjData(self.model)
        try:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        except Exception:
            pass
        mujoco.mj_forward(self.model, self.data)

    def _scene_center_and_distance(self):
        points = [np.array([0.0, 0.0, 0.85], dtype=float), self.robot_base_pos.astype(float)]
        points.extend([a.pos.astype(float) for a in self.placed_assets])

        pts = np.stack(points, axis=0)
        center = np.mean(pts, axis=0)
        span = np.max(pts, axis=0) - np.min(pts, axis=0)
        distance = float(max(np.linalg.norm(span) * 1.8, 2.0))
        return center, distance

    def _render_fixed_camera_image(self, camera_name: str, save_path: Path):
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"Camera {camera_name} not found")

        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id

        renderer = mujoco.Renderer(self.model, VIEW_IMAGE_WIDTH, VIEW_IMAGE_HEIGHT)
        try:
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            renderer._scene_option.sitegroup[:] = False
            renderer.update_scene(self.data)
            mujoco.mjv_updateCamera(self.model, self.data, camera, renderer._scene)
            image = renderer.render().astype(np.uint8)
        finally:
            renderer.close()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(save_path, image)

    def _render_free_view_image(self, azimuth: float, elevation: float, save_path: Path):
        center, distance = self._scene_center_and_distance()

        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = center
        camera.distance = distance
        camera.azimuth = float(azimuth)
        camera.elevation = float(elevation)

        renderer = mujoco.Renderer(self.model, VIEW_IMAGE_WIDTH, VIEW_IMAGE_HEIGHT)
        try:
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            renderer._scene_option.sitegroup[:] = False
            renderer.update_scene(self.data, camera=camera)
            image = renderer.render().astype(np.uint8)
        finally:
            renderer.close()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(save_path, image)

    def render_three_views(self):
        front_path = VIEW_LOG_DIR / "front.png"
        top_path = VIEW_LOG_DIR / "top.png"
        side_path = VIEW_LOG_DIR / "side.png"

        self._render_fixed_camera_image("table_cam_front", front_path)
        self._render_free_view_image(0.0, -89.0, top_path)
        self._render_free_view_image(90.0, -10.0, side_path)

        return str(front_path), str(top_path), str(side_path)


RUNTIME = SimpleRuntime()


# =========================
# Gradio 回调
# =========================
def refresh_views():
    return RUNTIME.render_three_views()


def load_scene(scene_file, scene_path_text, status_text):
    try:
        path = resolve_input_path(scene_file, scene_path_text)
        if path is None:
            path = DEFAULT_SCENE_JSON

        RUNTIME.load_scene_json(path)
        front, top, side = RUNTIME.render_three_views()
        status = f"Loaded JSON: {path}"
        return front, top, side, status
    except Exception as e:
        front, top, side = RUNTIME.render_three_views()
        return front, top, side, f"Load failed: {e}"


# =========================
# UI
# =========================
def build_app():
    front_default, top_default, side_default = RUNTIME.render_three_views()

    with gr.Blocks(title="EvoBody Simple Viewer") as demo:
        gr.Markdown("## EvoBody Simple Viewer")
        gr.Markdown("读取 JSON 场景并渲染三视图，不提供任何编辑功能。")

        with gr.Row():
            scene_file = gr.File(label="导入 Scene JSON", file_types=[".json"])
            scene_path = gr.Textbox(
                value="logs/exported_scene.json",
                label="JSON 路径（可直接填写，不上传也行）"
            )

        with gr.Row():
            load_btn = gr.Button("加载 JSON 场景", variant="primary")
            refresh_btn = gr.Button("刷新渲染")

        status_box = gr.Textbox(label="状态", value="", interactive=False)

        with gr.Row():
            front_image = gr.Image(value=front_default, label="Front View", type="filepath")
            top_image = gr.Image(value=top_default, label="Top View", type="filepath")
            side_image = gr.Image(value=side_default, label="Side View", type="filepath")

        load_btn.click(
            fn=load_scene,
            inputs=[scene_file, scene_path, status_box],
            outputs=[front_image, top_image, side_image, status_box],
        )

        refresh_btn.click(
            fn=lambda: (*refresh_views(), "Refreshed"),
            inputs=[],
            outputs=[front_image, top_image, side_image, status_box],
        )

    return demo


def main():
    app = build_app()
    app.queue()

    for port in [7862, 7863, 7864, 7865, 7870]:
        try:
            app.launch(
                server_name="localhost",
                server_port=port,
                theme=gr.themes.Soft(),
                allowed_paths=[
                    str(PROJECT_ROOT),
                    str(LOG_ROOT),
                    str(VIEW_LOG_DIR),
                    str(tempfile.gettempdir()),
                ],
            )
            return
        except OSError:
            continue

    raise RuntimeError("没有找到可用端口，请先关闭占用 786x 的 Gradio 进程。")


if __name__ == "__main__":
    main()