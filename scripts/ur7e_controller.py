"""
Example: control a UR7e with a 7D end-effector delta vector from a PC.

Vector format:
    [dx, dy, dz, droll, dpitch, dyaw, g]
where:
    - dx,dy,dz are TCP position increments in meters
    - droll,dpitch,dyaw are TCP orientation increments in radians
    - g is gripper command in [0.0, 1.0]
      0.0 -> fully open, 1.0 -> fully close

Interfaces used:
    - URScript socket on port 30003 for arm motion
    - URScript secondary socket on port 30002 for Robotiq URCap fallback
    - RTDE receive on port 30004 for reading current joints
    - Robotiq socket on port 63352 for direct gripper command (preferred)

Before running:
1) Put the robot in Remote Control mode (UR e-Series requirement for remote URScript).
2) Ensure networking from PC to robot is working.
3) Confirm your gripper accepts Robotiq text commands on port 63352.
    If not, provide Robotiq URScript definitions file and use URScript fallback.
4) Test in free-space first and keep E-Stop accessible.
"""

from __future__ import annotations

import socket
import time
import importlib
import math
import threading
from typing import Any, Iterable, Sequence

ROBOT_IP = "169.254.134.8"
URSCRIPT_PORT = 30003
URSCRIPT_SECONDARY_PORT = 30002
RTDE_PORT = 30004
ROBOTIQ_PORT = 63352


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class _RobotiqSocketClient:
    """Robotiq URCap socket client matching the known-working reference logic."""

    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout_s)
        self._sock.connect((self.host, self.port))
        self.activate()

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _send(self, cmd: bytes) -> str:
        if self._sock is None:
            raise RuntimeError("Robotiq socket is not connected.")
        with self._lock:
            self._sock.sendall(cmd)
            try:
                return self._sock.recv(1024).decode("ascii", errors="ignore").strip()
            except socket.timeout:
                return ""

    def send_line(self, cmd: str) -> str:
        return self._send((cmd.strip() + "\n").encode("ascii"))

    def activate(self) -> None:
        self._send(b"SET ACT 0\n")
        time.sleep(0.1)
        self._send(b"SET ACT 1\n")

        deadline = time.time() + 10.0
        while time.time() < deadline:
            resp = self._send(b"GET STA\n")
            if "STA 3" in resp:
                break
            time.sleep(0.1)

        self._send(b"SET GTO 1\n")

    def move(self, position: int, speed: int = 200, force: int = 150) -> None:
        position = max(0, min(255, int(position)))
        speed = max(0, min(255, int(speed)))
        force = max(0, min(255, int(force)))
        cmd = (
            f"SET POS {position}\n"
            f"SET SPE {speed}\n"
            f"SET FOR {force}\n"
            "SET GTO 1\n"
        ).encode("ascii")
        self._send(cmd)

    def get_var(self, var_name: str) -> int:
        resp = self.send_line(f"GET {var_name}")
        try:
            return int(resp.split()[-1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"Unexpected gripper response: {resp}") from exc


def _normalize_quaternion(quat_xyzw: Sequence[float]) -> list[float]:
    x, y, z, w = [float(v) for v in quat_xyzw]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / n, y / n, z / n, w / n]


def _quat_multiply(q1_xyzw: Sequence[float], q2_xyzw: Sequence[float]) -> list[float]:
    x1, y1, z1, w1 = _normalize_quaternion(q1_xyzw)
    x2, y2, z2, w2 = _normalize_quaternion(q2_xyzw)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return _normalize_quaternion([x, y, z, w])


def _rotvec_to_quat(rotvec_xyz: Sequence[float]) -> list[float]:
    rx, ry, rz = [float(v) for v in rotvec_xyz]
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    ax, ay, az = rx / theta, ry / theta, rz / theta
    half = 0.5 * theta
    s = math.sin(half)
    return [ax * s, ay * s, az * s, math.cos(half)]


def _euler_delta_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    qx = _rotvec_to_quat([roll, 0.0, 0.0])
    qy = _rotvec_to_quat([0.0, pitch, 0.0])
    qz = _rotvec_to_quat([0.0, 0.0, yaw])
    return _quat_multiply(qz, _quat_multiply(qy, qx))


def _quat_to_rotvec(quat_xyzw: Sequence[float]) -> list[float]:
    x, y, z, w = _normalize_quaternion(quat_xyzw)
    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return [0.0, 0.0, 0.0]

    angle = 2.0 * math.atan2(sin_half, w)
    ax, ay, az = x / sin_half, y / sin_half, z / sin_half
    return [ax * angle, ay * angle, az * angle]


def _quat_xyzw_to_wxyz(quat_xyzw: Sequence[float]) -> list[float]:
    x, y, z, w = _normalize_quaternion(quat_xyzw)
    return [w, x, y, z]


def _quat_wxyz_to_xyzw(quat_wxyz: Sequence[float]) -> list[float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    return _normalize_quaternion([x, y, z, w])


def _extract_pose(pose: Any, fallback_pos: Sequence[float], fallback_quat_wxyz: Sequence[float]) -> tuple[list[float], list[float]]:
    if isinstance(pose, dict):
        pos = pose.get("pos", pose.get("position", fallback_pos))
        quat = pose.get("quat", pose.get("quaternion", fallback_quat_wxyz))
    elif isinstance(pose, (list, tuple)) and len(pose) >= 2:
        pos, quat = pose[0], pose[1]
    else:
        pos, quat = fallback_pos, fallback_quat_wxyz
    return (
        [float(pos[0]), float(pos[1]), float(pos[2])],
        [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
    )


def _target_from_pose_or_offset(
    target_pose: Any,
    src_pos: Sequence[float],
    src_quat_wxyz: Sequence[float],
    direction_x: float,
    direction_y: float,
    direction_z: float,
) -> tuple[list[float], list[float]]:
    if target_pose is not None:
        return _extract_pose(target_pose, src_pos, src_quat_wxyz)
    return (
        [
            float(src_pos[0]) + float(direction_x),
            float(src_pos[1]) + float(direction_y),
            float(src_pos[2]) + float(direction_z),
        ],
        [float(src_quat_wxyz[0]), float(src_quat_wxyz[1]), float(src_quat_wxyz[2]), float(src_quat_wxyz[3])],
    )


class UR7eVectorController:
    def __init__(
        self,
        robot_ip: str = ROBOT_IP,
        urscript_port: int = URSCRIPT_PORT,
        urscript_secondary_port: int = URSCRIPT_SECONDARY_PORT,
        rtde_port: int = RTDE_PORT,
        robotiq_port: int = ROBOTIQ_PORT,
        robotiq_urscript_defs_path: str | None = None,
        timeout_s: float = 2.0,
        strict_gripper_connection: bool = False,
    ) -> None:
        self.robot_ip = robot_ip
        self.urscript_port = urscript_port
        self.urscript_secondary_port = urscript_secondary_port
        self.rtde_port = rtde_port
        self.robotiq_port = robotiq_port
        self.robotiq_urscript_defs_path = robotiq_urscript_defs_path
        self.timeout_s = timeout_s
        self.strict_gripper_connection = strict_gripper_connection

        self._ur_sock: socket.socket | None = None
        self._ur_secondary_sock: socket.socket | None = None
        self._gripper_client: _RobotiqSocketClient | None = None
        self._rtde_receive = None
        self._gripper_available = False
        self._gripper_warned = False
        self._last_gripper_open_ratio = 0.0
        self._gripper_backend = "none"
        self._robotiq_defs_cache: str | None = None

    def connect(self) -> None:
        try:
            rtde_module = importlib.import_module("rtde_receive")
            rtde_receive_interface = getattr(rtde_module, "RTDEReceiveInterface")
        except Exception as exc:
            raise ImportError(
                "Missing dependency 'ur-rtde'. Install with: pip install ur-rtde"
            ) from exc

        try:
            self._ur_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._ur_sock.settimeout(self.timeout_s)
            self._ur_sock.connect((self.robot_ip, self.urscript_port))

            self._ur_secondary_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._ur_secondary_sock.settimeout(self.timeout_s)
            self._ur_secondary_sock.connect((self.robot_ip, self.urscript_secondary_port))

            self._rtde_receive = rtde_receive_interface(self.robot_ip, self.rtde_port)
        except Exception:
            self.close()
            raise

        try:
            self._gripper_client = _RobotiqSocketClient(
                self.robot_ip,
                self.robotiq_port,
                self.timeout_s,
            )
            self._gripper_client.connect()
            self._gripper_available = True
            self._gripper_backend = "socket"
        except (socket.timeout, ConnectionError, OSError) as exc:
            if self._gripper_client is not None:
                try:
                    self._gripper_client.disconnect()
                finally:
                    self._gripper_client = None

            if self._try_enable_urscript_gripper_backend():
                self._gripper_available = True
                self._gripper_backend = "urscript"
                print(
                    "[WARN] Gripper socket 63352 unavailable; switched to URScript fallback "
                    "via port 30002 and Robotiq function definitions."
                )
                return

            self._gripper_available = False
            self._gripper_backend = "none"

            msg = (
                f"Gripper connection failed at {self.robot_ip}:{self.robotiq_port} ({exc}). "
                "Arm control is still available. "
                "If you need gripper control, confirm Robotiq URCap/socket service is enabled "
                "or set strict_gripper_connection=True to fail fast."
            )
            if self.strict_gripper_connection:
                self.close()
                raise TimeoutError(msg) from exc
            print(f"[WARN] {msg}")
        except Exception:
            self.close()
            raise

    def _try_enable_urscript_gripper_backend(self) -> bool:
        if self._ur_secondary_sock is None:
            return False
        if not self.robotiq_urscript_defs_path:
            return False

        try:
            defs = self._load_robotiq_defs()
            # Validate definitions by sending a no-op Robotiq query.
            script = (
                "def ext_gripper_ping():\n"
                f"{defs}\n"
                "  rq_is_gripper_activated()\n"
                "end\n"
                "ext_gripper_ping()\n"
            )
            self._send_urscript_secondary(script)
            return True
        except Exception as exc:
            print(f"[WARN] URScript gripper fallback unavailable: {exc}")
            return False

    def _load_robotiq_defs(self) -> str:
        if self._robotiq_defs_cache is not None:
            return self._robotiq_defs_cache

        if not self.robotiq_urscript_defs_path:
            raise RuntimeError("No Robotiq URScript definitions file is configured.")

        with open(self.robotiq_urscript_defs_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            raise RuntimeError("Robotiq URScript definitions file is empty.")

        self._robotiq_defs_cache = content
        return content

    def close(self) -> None:
        if self._ur_sock is not None:
            try:
                self._ur_sock.close()
            finally:
                self._ur_sock = None

        if self._ur_secondary_sock is not None:
            try:
                self._ur_secondary_sock.close()
            finally:
                self._ur_secondary_sock = None

        if self._gripper_client is not None:
            try:
                self._gripper_client.disconnect()
            finally:
                self._gripper_client = None

        if self._rtde_receive is not None:
            try:
                if hasattr(self._rtde_receive, "disconnect"):
                    self._rtde_receive.disconnect()
            finally:
                self._rtde_receive = None

        self._gripper_available = False
        self._gripper_backend = "none"

    def __enter__(self) -> "UR7eVectorController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_connected(self) -> None:
        if self._ur_sock is None:
            raise RuntimeError("Not connected. Call connect() first.")

    def _send_urscript(self, script_line: str) -> None:
        self._ensure_connected()
        assert self._ur_sock is not None
        payload = (script_line.strip() + "\n").encode("utf-8")
        self._ur_sock.sendall(payload)

    def _send_urscript_secondary(self, script_text: str) -> None:
        self._ensure_connected()
        if self._ur_secondary_sock is None:
            raise RuntimeError("Secondary URScript socket is not connected.")
        payload = script_text.strip() + "\n"
        self._ur_secondary_sock.sendall(payload.encode("utf-8"))

    def _send_gripper_cmd(self, cmd: str) -> None:
        self._ensure_connected()
        if not self._gripper_available or self._gripper_client is None:
            raise RuntimeError(
                "Gripper is unavailable: unable to send command. "
                "Check port 63352 service on robot controller."
            )
        self._gripper_client.send_line(cmd)

    def _get_gripper_var(self, var_name: str) -> int:
        self._ensure_connected()
        if not self._gripper_available or self._gripper_client is None:
            raise RuntimeError("Gripper is unavailable.")

        return self._gripper_client.get_var(var_name)

    def is_gripper_available(self) -> bool:
        return self._gripper_available

    def get_gripper_backend(self) -> str:
        return self._gripper_backend

    def activate_gripper(self) -> None:
        if self._gripper_backend == "urscript":
            defs = self._load_robotiq_defs()
            script = (
                "def ext_activate_gripper():\n"
                f"{defs}\n"
                "  rq_activate_and_wait()\n"
                "end\n"
                "ext_activate_gripper()\n"
            )
            self._send_urscript_secondary(script)
            return

        if self._gripper_client is None:
            raise RuntimeError("Gripper is unavailable.")
        self._gripper_client.activate()

    def move_joints(
        self,
        joints_rad: Sequence[float],
        acceleration: float = 1.2,
        velocity: float = 0.5,
        blend_radius: float = 0.0,
    ) -> None:
        if len(joints_rad) != 6:
            raise ValueError(f"Expected 6 joint values, got {len(joints_rad)}")

        q_str = ", ".join(f"{q:.6f}" for q in joints_rad)
        script = (
            f"movej([{q_str}], a={acceleration:.3f}, v={velocity:.3f}, r={blend_radius:.3f})"
        )
        self._send_urscript(script)

    def get_current_joints(self) -> list[float]:
        self._ensure_connected()
        if self._rtde_receive is None:
            raise RuntimeError("RTDE receive interface is not initialized.")

        actual_q = self._rtde_receive.getActualQ()
        if actual_q is None or len(actual_q) != 6:
            raise RuntimeError("Failed to read current joint positions from RTDE.")

        return [float(q) for q in actual_q]

    def get_current_tcp_pose(self) -> list[float]:
        self._ensure_connected()
        if self._rtde_receive is None:
            raise RuntimeError("RTDE receive interface is not initialized.")

        tcp_pose = self._rtde_receive.getActualTCPPose()
        if tcp_pose is None or len(tcp_pose) != 6:
            raise RuntimeError("Failed to read current TCP pose from RTDE.")

        return [float(v) for v in tcp_pose]

    def get_gripper_open_ratio(self) -> float:
        if not self._gripper_available:
            return self._last_gripper_open_ratio

        try:
            pos = self._get_gripper_var("POS")
            ratio = _clamp(float(pos) / 255.0, 0.0, 1.0)
            self._last_gripper_open_ratio = ratio
            return ratio
        except Exception:
            return self._last_gripper_open_ratio

    def get_current_ee_pose_vector(self) -> list[float]:
        """
        Returns [x, y, z, qx, qy, qz, qw, g].
        - Position unit: meters
        - Quaternion: xyzw
        - g in [0, 1]
        """
        tcp = self.get_current_tcp_pose()
        quat = _rotvec_to_quat(tcp[3:6])
        g = self.get_gripper_open_ratio()
        return [tcp[0], tcp[1], tcp[2], quat[0], quat[1], quat[2], quat[3], g]

    def set_gripper(self, g: float, speed: int = 255, force: int = 120) -> None:
        # Map g in [0,1] to Robotiq position [0,255].
        g = _clamp(float(g), 0.0, 1.0)
        position = int(round(g * 255.0))
        speed = max(0, min(255, int(speed)))
        force = max(0, min(255, int(force)))

        if self._gripper_backend == "urscript":
            defs = self._load_robotiq_defs()
            # urcap API uses position 0(open)~255(close); speed/force 0~255.
            script = (
                "def ext_set_gripper():\n"
                f"{defs}\n"
                f"  rq_set_speed_norm({speed})\n"
                f"  rq_set_force_norm({force})\n"
                f"  rq_move_and_wait({position})\n"
                "end\n"
                "ext_set_gripper()\n"
            )
            self._send_urscript_secondary(script)
            self._last_gripper_open_ratio = g
            return

        if self._gripper_client is None:
            raise RuntimeError("Gripper is unavailable.")
        self._gripper_client.move(position, speed=speed, force=force)
        self._last_gripper_open_ratio = g

    def ee_pose(self) -> tuple[list[float], list[float]]:
        """
        Real-runtime API: return current TCP pose as (pos_xyz, quat_wxyz).

        Generated real-world code should use quaternion order (w, x, y, z),
        matching the simulation atomic-code contract.
        """
        ee = self.get_current_ee_pose_vector()
        return [float(ee[0]), float(ee[1]), float(ee[2])], _quat_xyzw_to_wxyz(ee[3:7])

    def move_to(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        num_steps: int = 100,
    ) -> None:
        """
        Real-runtime API: move TCP to absolute world pose.

        Args:
            pos: [x, y, z] in meters.
            quat: [w, x, y, z]. If omitted, preserve current TCP orientation.
            num_steps: Accepted for compatibility with simulation code; real
                execution uses UR movel velocity/acceleration limits.
        """
        del num_steps
        if quat is None:
            _, quat = self.ee_pose()
        self.move_ee_pose(pos, _quat_wxyz_to_xyzw(quat), acceleration=0.4, velocity=0.10)

    def move_ee(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        droll: float = 0.0,
        dpitch: float = 0.0,
        dyaw: float = 0.0,
        steps: int = 100,
    ) -> None:
        """
        Real-runtime API: relative TCP delta.

        Rotation inputs are degrees to match the simulation runtime API.
        """
        del steps
        self.send_ee_delta_vector(
            [
                float(dx),
                float(dy),
                float(dz),
                math.radians(float(droll)),
                math.radians(float(dpitch)),
                math.radians(float(dyaw)),
                self.get_gripper_open_ratio(),
            ],
            acceleration=0.4,
            velocity=0.10,
        )

    def gripper_control(self, value: float, delay: int = 50) -> None:
        """
        Real-runtime API: Robotiq gripper command.

        `value` follows the simulation convention: 0=open, 255=closed.
        """
        self.set_gripper(_clamp(float(value) / 255.0, 0.0, 1.0))
        time.sleep(max(0.0, float(delay)) / 1000.0)

    def move_x(self, distance: float = 0.05, steps: int = 120) -> None:
        self.move_ee(dx=float(distance), dy=0.0, dz=0.0, steps=int(steps))

    def move_y(self, distance: float = 0.05, steps: int = 120) -> None:
        self.move_ee(dx=0.0, dy=float(distance), dz=0.0, steps=int(steps))

    def move_z(self, distance: float = 0.05, steps: int = 120) -> None:
        self.move_ee(dx=0.0, dy=0.0, dz=float(distance), steps=int(steps))

    def rotate_x(self, angle_deg: float = 10.0, steps: int = 120) -> None:
        self.move_ee(droll=float(angle_deg), steps=int(steps))

    def rotate_y(self, angle_deg: float = 10.0, steps: int = 120) -> None:
        self.move_ee(dpitch=float(angle_deg), steps=int(steps))

    def rotate_z(self, angle_deg: float = 10.0, steps: int = 120) -> None:
        self.move_ee(dyaw=float(angle_deg), steps=int(steps))

    def pick_and_place(
        self,
        object_pose: Any = None,
        target_pose: Any = None,
        direction_x: float = 0.0,
        direction_y: float = 0.0,
        direction_z: float = 0.0,
        approach_height: float = 0.08,
        lift_height: float = 0.10,
        grasp_value: float = 255,
        release_value: float = 0,
        move_steps: int = 120,
        grip_delay: int = 80,
    ) -> None:
        cur_pos, cur_quat = self.ee_pose()
        src_pos, src_quat = _extract_pose(object_pose, cur_pos, cur_quat)
        dst_pos, _ = _target_from_pose_or_offset(
            target_pose,
            src_pos,
            src_quat,
            direction_x,
            direction_y,
            direction_z,
        )
        carry_z = src_pos[2] + float(lift_height)
        retreat_z = max(carry_z, dst_pos[2] + float(approach_height))

        self.gripper_control(float(release_value), delay=int(grip_delay))
        self.move_to([src_pos[0], src_pos[1], src_pos[2] + float(approach_height)], src_quat, num_steps=int(move_steps))
        self.move_to(src_pos, src_quat, num_steps=int(move_steps))
        self.gripper_control(float(grasp_value), delay=int(grip_delay))
        self.move_to([src_pos[0], src_pos[1], carry_z], src_quat, num_steps=int(move_steps))
        self.move_to([dst_pos[0], dst_pos[1], carry_z], src_quat, num_steps=int(move_steps))
        self.move_to(dst_pos, src_quat, num_steps=int(move_steps))
        self.gripper_control(float(release_value), delay=int(grip_delay))
        self.move_to([dst_pos[0], dst_pos[1], retreat_z], src_quat, num_steps=int(move_steps))

    def pick_place(self, *args: Any, **kwargs: Any) -> None:
        self.pick_and_place(*args, **kwargs)

    def push(
        self,
        target_pose: Any = None,
        object_pose: Any = None,
        push_distance: float = 0.05,
        approach_height: float = 0.05,
        grasp_value: float = 255,
        move_steps: int = 100,
        grip_delay: int = 60,
    ) -> None:
        cur_pos, cur_quat = self.ee_pose()
        tgt_pos, tgt_quat = _extract_pose(target_pose if target_pose is not None else object_pose, cur_pos, cur_quat)
        if float(approach_height) > 1e-9:
            self.move_to([tgt_pos[0], tgt_pos[1], tgt_pos[2] + float(approach_height)], tgt_quat, num_steps=int(move_steps))
        self.move_to(tgt_pos, tgt_quat, num_steps=int(move_steps))
        self.gripper_control(float(grasp_value), delay=int(grip_delay))
        self.move_ee(dz=abs(float(push_distance)), steps=int(move_steps))

    def pull(
        self,
        target_pose: Any = None,
        object_pose: Any = None,
        pull_distance: float = 0.05,
        approach_height: float = 0.05,
        grasp_value: float = 255,
        release_value: float = 0,
        move_steps: int = 100,
        grip_delay: int = 60,
    ) -> None:
        cur_pos, cur_quat = self.ee_pose()
        tgt_pos, tgt_quat = _extract_pose(target_pose if target_pose is not None else object_pose, cur_pos, cur_quat)
        if float(approach_height) > 1e-9:
            self.move_to([tgt_pos[0], tgt_pos[1], tgt_pos[2] + float(approach_height)], tgt_quat, num_steps=int(move_steps))
        self.move_to(tgt_pos, tgt_quat, num_steps=int(move_steps))
        self.gripper_control(float(grasp_value), delay=int(grip_delay))
        self.move_ee(dz=-abs(float(pull_distance)), steps=int(move_steps))
        self.gripper_control(float(release_value), delay=int(grip_delay))

    def press(
        self,
        object_pose: Any = None,
        direction_x: float = 0.0,
        direction_y: float = 0.0,
        direction_z: float = -0.03,
        grasp_value: float = 255,
        move_steps: int = 90,
        grip_delay: int = 60,
    ) -> None:
        del object_pose
        self.gripper_control(float(grasp_value), delay=int(grip_delay))
        self.move_ee(dx=float(direction_x), dy=float(direction_y), dz=float(direction_z), steps=int(move_steps))

    def open(
        self,
        grasp_pose: Any = None,
        rotation_radius: float = 0.08,
        rotation_angle_deg: float = 45.0,
        grasp_value: float = 255,
        release_value: float = 0,
        move_steps: int = 100,
        grip_delay: int = 80,
        **kwargs: Any,
    ) -> None:
        del rotation_radius, kwargs
        cur_pos, cur_quat = self.ee_pose()
        gpos, gquat = _extract_pose(grasp_pose, cur_pos, cur_quat)
        self.move_to(gpos, gquat, num_steps=int(move_steps))
        self.gripper_control(float(grasp_value), delay=int(grip_delay))
        self.rotate_y(float(rotation_angle_deg), steps=int(move_steps))
        self.gripper_control(float(release_value), delay=int(grip_delay))

    def close_articulation(
        self,
        grasp_pose: Any = None,
        rotation_radius: float = 0.08,
        rotation_angle_deg: float = -45.0,
        grasp_value: float = 255,
        move_steps: int = 100,
        grip_delay: int = 80,
        **kwargs: Any,
    ) -> None:
        del rotation_radius, kwargs
        cur_pos, cur_quat = self.ee_pose()
        gpos, gquat = _extract_pose(grasp_pose, cur_pos, cur_quat)
        self.move_to(gpos, gquat, num_steps=int(move_steps))
        self.gripper_control(float(grasp_value), delay=int(grip_delay))
        self.rotate_y(float(rotation_angle_deg), steps=int(move_steps))

    def pour(
        self,
        object_pose: Any = None,
        target_pose: Any = None,
        direction_x: float = 0.08,
        direction_y: float = 0.0,
        direction_z: float = 0.03,
        rot_x: float = 0.0,
        rot_y: float = 60.0,
        rot_z: float = 0.0,
        approach_height: float = 0.08,
        lift_height: float = 0.12,
        grasp_value: float = 255,
        release_value: float = 0,
        move_steps: int = 120,
        grip_delay: int = 80,
    ) -> None:
        cur_pos, cur_quat = self.ee_pose()
        src_pos, src_quat = _extract_pose(object_pose, cur_pos, cur_quat)
        dst_pos, dst_quat = _target_from_pose_or_offset(
            target_pose,
            src_pos,
            src_quat,
            direction_x,
            direction_y,
            direction_z,
        )
        self.move_to([src_pos[0], src_pos[1], src_pos[2] + float(approach_height)], src_quat, num_steps=int(move_steps))
        self.move_to(src_pos, src_quat, num_steps=int(move_steps))
        self.gripper_control(float(grasp_value), delay=int(grip_delay))
        self.move_to([src_pos[0], src_pos[1], src_pos[2] + float(lift_height)], src_quat, num_steps=int(move_steps))
        self.move_to([dst_pos[0], dst_pos[1], dst_pos[2] + float(approach_height)], dst_quat, num_steps=int(move_steps))
        self.move_to(dst_pos, dst_quat, num_steps=int(move_steps))
        self.move_ee(droll=float(rot_x), dpitch=float(rot_y), dyaw=float(rot_z), steps=int(move_steps))
        self.gripper_control(float(release_value), delay=int(grip_delay))

    def move_ee_pose(
        self,
        position_xyz: Sequence[float],
        quat_xyzw: Sequence[float],
        acceleration: float = 0.4,
        velocity: float = 0.15,
        blend_radius: float = 0.0,
    ) -> None:
        if len(position_xyz) != 3:
            raise ValueError(f"Expected 3 position values, got {len(position_xyz)}")
        if len(quat_xyzw) != 4:
            raise ValueError(f"Expected 4 quaternion values, got {len(quat_xyzw)}")

        rotvec = _quat_to_rotvec(quat_xyzw)
        x, y, z = [float(v) for v in position_xyz]
        rx, ry, rz = rotvec
        script = (
            "movel(p["
            f"{x:.6f}, {y:.6f}, {z:.6f}, {rx:.6f}, {ry:.6f}, {rz:.6f}"
            f"], a={acceleration:.3f}, v={velocity:.3f}, r={blend_radius:.3f})"
        )
        self._send_urscript(script)

    def send_vector(
        self,
        vector7: Iterable[float],
        acceleration: float = 1.2,
        velocity: float = 0.5,
        wait_after_arm_s: float = 0.2,
    ) -> tuple[list[float], list[float]]:
        values = [float(x) for x in vector7]
        if len(values) != 7:
            raise ValueError(f"Expected 7 values [dq0..dq5,g], got {len(values)}")

        delta_joints = values[:6]
        gripper = values[6]

        current_joints = self.get_current_joints()
        target_joints = [cur + dq for cur, dq in zip(current_joints, delta_joints)]

        self.move_joints(target_joints, acceleration=acceleration, velocity=velocity)
        # Small delay so arm and gripper commands do not saturate interfaces.
        time.sleep(wait_after_arm_s)
        if self._gripper_available:
            self.set_gripper(gripper)
        elif not self._gripper_warned:
            print(
                "[WARN] Gripper command skipped because gripper socket is unavailable."
            )
            self._gripper_warned = True

        # Return both vectors so caller can log/verify before and after planning.
        return current_joints, target_joints

    def send_ee_delta_vector(
        self,
        delta7: Iterable[float],
        acceleration: float = 0.4,
        velocity: float = 0.15,
        wait_after_arm_s: float = 0.2,
    ) -> tuple[list[float], list[float]]:
        """
        EE delta vector format:
            [dx, dy, dz, droll, dpitch, dyaw, g]
        where:
            - dx,dy,dz are deltas applied to current TCP position
            - droll,dpitch,dyaw are Euler-angle orientation deltas
            - quaternion delta composes in base frame: q_target = dq * q_current
            - g is an absolute gripper command in [0, 1] range after clamping

        Returns:
            (current_ee, target_ee), each as [x, y, z, qx, qy, qz, qw, g]
        """
        values = [float(x) for x in delta7]
        if len(values) != 7:
            raise ValueError(f"Expected 7 values [dx,dy,dz,droll,dpitch,dyaw,g], got {len(values)}")

        current_ee = self.get_current_ee_pose_vector()
        cur_pos = current_ee[:3]
        cur_quat = current_ee[3:7]

        delta_pos = values[:3]
        droll, dpitch, dyaw = values[3:6]
        target_g = _clamp(values[6], 0.0, 1.0)

        target_pos = [p + dp for p, dp in zip(cur_pos, delta_pos)]
        delta_quat = _euler_delta_to_quat(droll, dpitch, dyaw)
        target_quat = _quat_multiply(delta_quat, cur_quat)

        self.move_ee_pose(
            target_pos,
            target_quat,
            acceleration=acceleration,
            velocity=velocity,
        )

        time.sleep(wait_after_arm_s)
        if self._gripper_available:
            self.set_gripper(target_g)
        elif not self._gripper_warned:
            print("[WARN] Gripper command skipped because gripper socket is unavailable.")
            self._gripper_warned = True

        target_ee = [
            target_pos[0],
            target_pos[1],
            target_pos[2],
            target_quat[0],
            target_quat[1],
            target_quat[2],
            target_quat[3],
            target_g,
        ]
        return current_ee, target_ee


def demo() -> None:
    # EE delta command: [dx, dy, dz, droll, dpitch, dyaw, g]
    command_sequence = [
        [0.00, 0.00, 0.02, 0.0, 0.0, 0.0, 0.0],
        [0.00, 0.00, -0.02, 0.0, 0.0, 0.0, 0.0],
    ]

    # If 63352 is blocked, provide local Robotiq definitions script path for URScript fallback.
    robotiq_defs_path = None

    with UR7eVectorController(
        robot_ip=ROBOT_IP,
        robotiq_urscript_defs_path=robotiq_defs_path,
        strict_gripper_connection=False,
    ) as controller:
        if controller.is_gripper_available():
            print(f"Gripper backend: {controller.get_gripper_backend()}")
            controller.activate_gripper()

        current_ee = controller.get_current_ee_pose_vector()
        print(f"Current EE pose vector: {current_ee}")

        for idx, vec in enumerate(command_sequence, start=1):
            print(f"Sending EE delta vector #{idx}: {vec}")
            current_pose, target_pose = controller.send_ee_delta_vector(
                vec, acceleration=0.4, velocity=0.1
            )
            print(f"Current EE pose before send: {current_pose}")
            print(f"Target EE pose after delta: {target_pose}")
            time.sleep(1.2)


def make_real_runtime_api(controller: UR7eVectorController) -> dict[str, Any]:
    """Return the restricted real-robot API dictionary for generated code."""
    return {
        "ee_pose": controller.ee_pose,
        "move_to": controller.move_to,
        "move_ee": controller.move_ee,
        "gripper_control": controller.gripper_control,
        "set_gripper": lambda value: controller.gripper_control(value, delay=1),
        "move_x": controller.move_x,
        "move_y": controller.move_y,
        "move_z": controller.move_z,
        "rotate_x": controller.rotate_x,
        "rotate_y": controller.rotate_y,
        "rotate_z": controller.rotate_z,
        "pick_and_place": controller.pick_and_place,
        "pick_place": controller.pick_place,
        "push": controller.push,
        "pull": controller.pull,
        "press": controller.press,
        "open": controller.open,
        "close": controller.close_articulation,
        "pour": controller.pour,
        "sleep": time.sleep,
        "print": print,
    }


if __name__ == "__main__":
    demo()
