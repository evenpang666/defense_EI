"""
Example: control a UR7e with a 7D vector from a PC.

Vector format:
    [dq0, dq1, dq2, dq3, dq4, dq5, g]
where:
    - dq0..dq5 are joint increments in radians
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
from typing import Iterable, Sequence

ROBOT_IP = "169.254.134.8"
URSCRIPT_PORT = 30003
URSCRIPT_SECONDARY_PORT = 30002
RTDE_PORT = 30004
ROBOTIQ_PORT = 63352


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def _quat_to_rotvec(quat_xyzw: Sequence[float]) -> list[float]:
    x, y, z, w = _normalize_quaternion(quat_xyzw)
    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return [0.0, 0.0, 0.0]

    angle = 2.0 * math.atan2(sin_half, w)
    ax, ay, az = x / sin_half, y / sin_half, z / sin_half
    return [ax * angle, ay * angle, az * angle]


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
        self._gripper_sock: socket.socket | None = None
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
            self._gripper_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._gripper_sock.settimeout(self.timeout_s)
            self._gripper_sock.connect((self.robot_ip, self.robotiq_port))
            self._gripper_available = True
            self._gripper_backend = "socket"
        except (socket.timeout, ConnectionError, OSError) as exc:
            if self._gripper_sock is not None:
                try:
                    self._gripper_sock.close()
                finally:
                    self._gripper_sock = None

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

        if self._gripper_sock is not None:
            try:
                self._gripper_sock.close()
            finally:
                self._gripper_sock = None

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
        if not self._gripper_available or self._gripper_sock is None:
            raise RuntimeError(
                "Gripper is unavailable: unable to send command. "
                "Check port 63352 service on robot controller."
            )
        assert self._gripper_sock is not None
        self._gripper_sock.sendall((cmd.strip() + "\n").encode("ascii"))

    def _get_gripper_var(self, var_name: str) -> int:
        self._ensure_connected()
        if not self._gripper_available or self._gripper_sock is None:
            raise RuntimeError("Gripper is unavailable.")

        cmd = f"GET {var_name}\n"
        self._gripper_sock.sendall(cmd.encode("ascii"))
        data = self._gripper_sock.recv(1024).decode("ascii", errors="ignore").strip()
        parts = data.split()
        if len(parts) < 2 or parts[0] != var_name:
            raise RuntimeError(f"Unexpected gripper response: {data}")
        return int(parts[1])

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

        self._send_gripper_cmd("SET ACT 0")
        time.sleep(0.05)
        self._send_gripper_cmd("SET ACT 1")
        self._send_gripper_cmd("SET GTO 1")
        self._send_gripper_cmd("SET SPE 255")
        self._send_gripper_cmd("SET FOR 120")
        time.sleep(0.3)

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

        self._send_gripper_cmd("SET ACT 1")
        self._send_gripper_cmd("SET MOD 1")
        self._send_gripper_cmd(f"SET SPE {speed}")
        self._send_gripper_cmd(f"SET FOR {force}")
        self._send_gripper_cmd(f"SET POS {position}")
        self._send_gripper_cmd("SET GTO 1")
        self._last_gripper_open_ratio = g

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
        delta8: Iterable[float],
        acceleration: float = 0.4,
        velocity: float = 0.15,
        wait_after_arm_s: float = 0.2,
    ) -> tuple[list[float], list[float]]:
        """
        EE delta vector format:
            [dx, dy, dz, dqx, dqy, dqz, dqw, dg]
        where:
            - d* are deltas applied to current EE pose
            - quaternion delta composes in base frame: q_target = dq * q_current
            - dg is gripper opening delta in [0, 1] range after clamping

        Returns:
            (current_ee, target_ee), each as [x, y, z, qx, qy, qz, qw, g]
        """
        values = [float(x) for x in delta8]
        if len(values) != 8:
            raise ValueError(f"Expected 8 values [dx,dy,dz,dqx,dqy,dqz,dqw,dg], got {len(values)}")

        current_ee = self.get_current_ee_pose_vector()
        cur_pos = current_ee[:3]
        cur_quat = current_ee[3:7]
        cur_g = current_ee[7]

        delta_pos = values[:3]
        delta_quat = values[3:7]
        delta_g = values[7]

        if sum(abs(v) for v in delta_quat) < 1e-12:
            delta_quat = [0.0, 0.0, 0.0, 1.0]

        target_pos = [p + dp for p, dp in zip(cur_pos, delta_pos)]
        target_quat = _quat_multiply(delta_quat, cur_quat)
        target_g = _clamp(cur_g + delta_g, 0.0, 1.0)

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
    # EE delta command: [dx, dy, dz, dqx, dqy, dqz, dqw, dg]
    command_sequence = [
        [0.00, 0.00, 0.02, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.00, 0.00, -0.02, 0.0, 0.0, 0.0, 1.0, 0.0],
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


if __name__ == "__main__":
    demo()
