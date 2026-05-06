import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pipeline = project_root / "scripts" / "libero" / "pipeline_libero_long_lora_moe.py"
    cmd = [sys.executable, str(pipeline), *sys.argv[1:], "--stage", "train"]
    rc = subprocess.run(cmd, cwd=str(project_root), check=False).returncode
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
