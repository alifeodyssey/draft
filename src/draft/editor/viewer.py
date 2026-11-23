"""The 3D half of the editor: a viser scene showing one generated model.

Owns the scene and physics loop and exposes the joints as data; the joint
sliders and simulate toggle live on the editor page (`editor/web/`), which runs
this class in the same process.
"""

from __future__ import annotations

import threading
import time
import math
from pathlib import Path
from typing import Any

import mujoco
import viser
from mjviser.scene import ViserMujocoScene

from draft.generation import mjcf_assets


class Viewer3D:
    """A viser server showing one MuJoCo model. The editor page drives it."""

    def __init__(self, port: int = 8080, verbose: bool = True) -> None:
        self.server = viser.ViserServer(port=port)
        # viser's own panel cannot be hidden; `floating` overlays the canvas
        # instead of taking a column from it.
        self.server.gui.configure_theme(control_layout="floating",
                                        show_logo=False, show_share_button=False)
        # The bound port: viser moves to the next free one if `port` is taken.
        self.port = self.server.get_port()
        if verbose:
            print(f"[viewer] 3D scene on http://localhost:{self.port}", flush=True)

        self._mj_model: mujoco.MjModel | None = None
        self._mj_data: mujoco.MjData | None = None
        self._scene: ViserMujocoScene | None = None
        self._scene_lock = threading.Lock()
        self._simulate = False
        #: One entry per movable joint: name, limits, and its qpos address.
        #: Handed to the page, which draws the sliders.
        self._joints: list[dict[str, Any]] = []
        self._preview_velocities: dict[str, float] = {}
        self._status = "Ready."

    # ── what the page drives ─────────────────────────────────────────────────

    def set_status(self, msg: str) -> None:
        self._status = msg

    @property
    def simulate(self) -> bool:
        return self._simulate

    @property
    def has_standing_pose(self) -> bool:
        return (self._mj_model is not None and mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_KEY, 'standing') >= 0)

    @staticmethod
    def _reset_data(model: mujoco.MjModel, data: mujoco.MjData,
                    *, zero: bool = False) -> None:
        key = (-1 if zero else mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, 'standing'))
        if key >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

    def set_simulate(self, on: bool) -> None:
        """Run physics, or hold the slider pose (resetting on the way out)."""
        with self._scene_lock:
            self._simulate = bool(on)
            self._clear_preview()
        if not self._simulate:
            self.reset_pose()

    def joints(self) -> list[dict]:
        """The movable joints of the loaded model, with their current angles."""
        out = []
        with self._scene_lock:
            for j in self._joints:
                q = (float(self._mj_data.qpos[j["qadr"]])
                     if self._mj_data is not None else 0.0)
                out.append({**j, "value": q,
                            "preview_velocity": getattr(self, '_preview_velocities', {}).get(
                                j['name'], 0.0)})
        return out

    def set_joint(self, qadr: int, value: float) -> None:
        """Pose one joint. Ignored while physics is running, which owns qpos."""
        if self._simulate:
            return
        with self._scene_lock:
            if self._mj_data is None or self._mj_model is None:
                return
            if not 0 <= qadr < len(self._mj_data.qpos):
                return
            self._mj_data.qpos[qadr] = float(value)
            mujoco.mj_forward(self._mj_model, self._mj_data)
            if self._scene:
                self._scene.update_from_mjdata(self._mj_data)

    def _clear_preview(self) -> None:
        """Called under the scene lock; preview momentum never enters physics."""
        self._preview_velocities = {}
        if self._mj_data is not None:
            for joint in getattr(self, '_joints', []):
                if joint.get('continuous'):
                    self._mj_data.qvel[joint['vadr']] = 0.0

    def stop_preview(self) -> None:
        with self._scene_lock:
            self._clear_preview()

    def set_preview_velocity(self, name: str, velocity: float) -> None:
        """Kinematic preview only: independent unlimited hinges, in rad/s."""
        if not math.isfinite(velocity):
            raise ValueError('preview velocity must be finite')
        with self._scene_lock:
            if self._simulate:
                return
            joint = next((j for j in self._joints if j['name'] == name), None)
            if joint is None or not joint.get('continuous'):
                raise ValueError(f'{name}: preview needs an unlimited hinge')
            limit = joint.get('velocity_limit')
            if limit is None or abs(velocity) > limit:
                raise ValueError(f'{name}: preview speed exceeds velocity limit {limit}')
            self._preview_velocities[name] = velocity

    def _step_preview(self, dt: float) -> None:
        """Advance raw qpos without wrapping; the page wraps its display only."""
        if self._simulate or self._mj_model is None or self._mj_data is None:
            return
        changed = False
        for j in self._joints:
            velocity = self._preview_velocities.get(j['name'], 0.0)
            if j.get('continuous'):
                self._mj_data.qvel[j['vadr']] = velocity
            if velocity:
                self._mj_data.qpos[j['qadr']] += velocity * dt
                changed = True
        if changed:
            mujoco.mj_forward(self._mj_model, self._mj_data)

    # ── model ─────────────────────────────────────────────────────────────────

    def load_model(self, mjcf_path: Path) -> bool:
        """Replace the 3D scene with a freshly-generated MJCF, in place."""
        scene_xml = Path(mjcf_path).parent / "scene.xml"
        load_path = scene_xml if scene_xml.is_file() else Path(mjcf_path)
        try:
            # Not `from_xml_path`: MuJoCo caches meshes by path, and every build
            # rewrites the same folder.
            model = mjcf_assets.load_model(load_path)
            data = mujoco.MjData(model)
            self._reset_data(model, data)
            with self._scene_lock:
                self.server.scene.reset()
                self._scene = ViserMujocoScene(self.server, model, num_envs=1)
                self._mj_model = model
                self._mj_data = data
                self._preview_velocities = {}
                self._scene.update_from_mjdata(data)
        except Exception as exc:
            self.set_status(f"**Error loading model:** {exc}")
            return False
        self._scan_joints()
        return True

    def reset_pose(self, *, zero: bool = False) -> None:
        with self._scene_lock:
            if self._mj_model is not None and self._mj_data is not None:
                self._clear_preview()
                self._reset_data(self._mj_model, self._mj_data, zero=zero)
                if self._scene:
                    self._scene.update_from_mjdata(self._mj_data)

    # ── joints ────────────────────────────────────────────────────────────────

    def _scan_joints(self) -> None:
        """Record the movable joints of the freshly-loaded model."""
        self._joints = []
        self._preview_velocities = {}
        model = self._mj_model
        if model is None:
            return
        HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
        SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
        for jid in range(model.njnt):
            jtype = int(model.jnt_type[jid])
            if jtype not in (HINGE, SLIDE):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
            limited = bool(model.jnt_limited[jid])
            continuous = jtype == HINGE and not limited
            if limited:
                lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
            elif continuous:
                lo = hi = None
            else:
                lo, hi = -1.0, 1.0
            numeric = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC,
                                        f'{name}_velocity_limit')
            speed = (float(model.numeric_data[model.numeric_adr[numeric]])
                     if numeric >= 0 and model.numeric_size[numeric] == 1 else None)
            if speed is not None and (not math.isfinite(speed) or speed <= 0):
                speed = None
            self._joints.append({
                "name": name, "lo": lo, "hi": hi,
                "qadr": int(model.jnt_qposadr[jid]),
                "vadr": int(model.jnt_dofadr[jid]),
                "limited": limited, "continuous": continuous,
                "velocity_limit": speed,
                "unit": "rad" if jtype == HINGE else "m",
            })

    # ── loop ──────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Run the render/physics loop on a daemon thread."""
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        previous = time.monotonic()
        while True:
            now = time.monotonic()
            elapsed, previous = min(now - previous, 0.1), now
            with self._scene_lock:
                if self._scene and self._mj_data and self._mj_model:
                    if self._simulate:
                        mujoco.mj_step(self._mj_model, self._mj_data)
                    else:
                        self._step_preview(elapsed)
                    self._scene.update_from_mjdata(self._mj_data)
            dt = (self._mj_model.opt.timestep
                  if (self._simulate and self._mj_model) else 1.0 / 60.0)
            time.sleep(dt)
