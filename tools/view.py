"""Hot-reload viewer: rebuilds the model whenever a file under model/ is saved.

    python -m tools.view                       # uses model.generator:build
    python -m tools.view model.leg:build_test  # any module:function returning MjSpec/MjModel
    python -m tools.view path/to/robot.xml
    python -m tools.view --frozen              # kinematics only (no stepping), with a joint slider window

If a rebuild fails, the traceback is printed and the last good model stays open.
Resets to the "stand" keyframe if the model has one.

In --frozen mode a separate "Joints" window has one slider per hinge or slide
joint. The viewer's own Joint panel can't be edited in passive mode, so use this
instead. Slider values are kept by joint name across rebuilds.
"""

import argparse
import math
import time
import tkinter as tk
import traceback
from pathlib import Path

import mujoco
import mujoco.viewer

from tools.loader import DEFAULT_SOURCE, GENERATED_DIR, ROOT, load

WATCH_DIR = ROOT / "model"
WATCH_SUFFIXES = {".py", ".yaml", ".yml", ".xml"}
POLL_SECONDS = 0.5


def snapshot() -> dict[Path, float]:
    files = {}
    for p in WATCH_DIR.rglob("*"):
        if p.suffix in WATCH_SUFFIXES and GENERATED_DIR not in p.parents:
            files[p] = p.stat().st_mtime
    return files


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


class JointSliders:
    """Tk window with one slider per hinge or slide joint, keyed by joint name."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Joints")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.open = True
        self.frame = None
        self.sliders: dict[str, tuple[int, tk.DoubleVar]] = {}  # name -> (qpos address, value)
        self.initial: dict[str, float] = {}

    def rebuild(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Make sliders for a new model, keeping values of joints that still exist."""
        if not self.open:
            return
        kept = {name: var.get() for name, (_, var) in self.sliders.items()}
        if self.frame is not None:
            self.frame.destroy()
        self.frame = tk.Frame(self.root, padx=8, pady=8)
        self.frame.pack(fill="both", expand=True)
        self.sliders, self.initial = {}, {}

        row = 0
        for i in range(model.njnt):
            if model.jnt_type[i] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint{i}"
            adr = model.jnt_qposadr[i]
            lo, hi = model.jnt_range[i] if model.jnt_limited[i] else (-math.pi, math.pi)
            self.initial[name] = float(data.qpos[adr])
            var = tk.DoubleVar(value=kept.get(name, self.initial[name]))
            tk.Label(self.frame, text=name, anchor="w", width=16).grid(row=row, column=0, sticky="sw")
            tk.Scale(self.frame, variable=var, from_=lo, to=hi, resolution=0.01, orient="horizontal",
                     length=280).grid(row=row, column=1)
            self.sliders[name] = (adr, var)
            row += 1

        buttons = tk.Frame(self.frame, pady=6)
        buttons.grid(row=row, column=0, columnspan=2)
        tk.Button(buttons, text="Zero", command=lambda: self._set_all(lambda n: 0.0)).pack(side="left", padx=4)
        tk.Button(buttons, text="Reset to keyframe", command=lambda: self._set_all(self.initial.get)).pack(side="left", padx=4)

    def apply(self, data: mujoco.MjData) -> None:
        for adr, var in self.sliders.values():
            data.qpos[adr] = var.get()

    def update(self) -> None:
        """Process Tk events; call this from the main loop."""
        if self.open:
            try:
                self.root.update()
            except tk.TclError:
                self.open = False

    def close(self) -> None:
        if self.open:
            self.root.destroy()
            self.open = False

    def _set_all(self, value_for) -> None:
        for name, (_, var) in self.sliders.items():
            var.set(value_for(name))

    def _on_close(self) -> None:
        # Closing the slider window keeps the viewer open, with the last pose held.
        self.root.destroy()
        self.open = False


def try_load(source: str, reload_modules: bool) -> mujoco.MjModel | None:
    try:
        return load(source, reload_modules=reload_modules)
    except Exception:
        traceback.print_exc()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default=DEFAULT_SOURCE)
    parser.add_argument("--frozen", action="store_true", help="don't step physics, only mj_forward")
    args = parser.parse_args()

    model = try_load(args.source, reload_modules=False)
    if model is None:
        print("Fix the error above and save; waiting for changes...")
    while model is None:
        before = snapshot()
        while snapshot() == before:
            time.sleep(POLL_SECONDS)
        model = try_load(args.source, reload_modules=True)

    sliders = JointSliders() if args.frozen else None
    cam_state = None
    while True:
        data = mujoco.MjData(model)
        reset(model, data)
        print(f"Loaded: nbody={model.nbody} njnt={model.njnt} nu={model.nu}")
        if sliders is not None:
            sliders.rebuild(model, data)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            if cam_state is not None:
                viewer.cam.lookat[:], viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam_state

            files = snapshot()
            last_poll = time.time()
            new_model = None
            while viewer.is_running() and new_model is None:
                step_start = time.time()
                with viewer.lock():
                    if args.frozen:
                        if sliders is not None and sliders.open:
                            sliders.apply(data)
                        mujoco.mj_forward(model, data)
                    else:
                        mujoco.mj_step(model, data)
                viewer.sync()
                if sliders is not None:
                    sliders.update()

                if time.time() - last_poll > POLL_SECONDS:
                    last_poll = time.time()
                    current = snapshot()
                    if current != files:
                        files = current
                        print("Change detected, rebuilding...")
                        new_model = try_load(args.source, reload_modules=True)

                # Run at roughly real time.
                sleep = model.opt.timestep - (time.time() - step_start)
                if sleep > 0:
                    time.sleep(sleep)

            if new_model is None:  # window closed by the user
                if sliders is not None:
                    sliders.close()
                return
            cam_state = (viewer.cam.lookat.copy(), viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation)
        model = new_model


if __name__ == "__main__":
    main()
