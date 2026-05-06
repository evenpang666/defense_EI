import mujoco
import mujoco.viewer
import argparse


def _try_load_optional_mjlab_plugin() -> None:
    loader = getattr(mujoco, "mj_loadPluginLibrary", None)
    if not callable(loader):
        return
    for plugin_path in ("./libmjlab.so.3.3.0", "./mjlab.dll", "./libmjlab.dylib"):
        try:
            loader(plugin_path)
            print(f"[test_xml] loaded plugin: {plugin_path}")
            return
        except Exception:
            continue
    print("[test_xml] optional mjlab plugin not found; continuing without it")

if __name__ == "__main__":
    _try_load_optional_mjlab_plugin()
    # add args about the path of xml
    parser = argparse.ArgumentParser(description="Load and visualize MuJoCo XML model")
    parser.add_argument("--xml", required=True, type=str, help="Path to the XML model file")
    args = parser.parse_args()
    
    # 加载 desk.xml 模型
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # 启动交互式查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 步进仿真
            mujoco.mj_step(model, data)
            # 同步查看器
            viewer.sync()