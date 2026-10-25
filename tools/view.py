"""Hot-reload viewer: rebuilds the model whenever a file under model/ is saved.

    python -m tools.view                       # uses model.generator:build
    python -m tools.view model.leg:build_test  # any module:function returning MjSpec/MjModel
    python -m tools.view path/to/robot.xml
    python -m tools.view --frozen              # kinematics only (no stepping), for checking geometry

If a rebuild fails, the traceback is printed and the last good model stays open.
Resets to the "stand" keyframe if the model has one.
"""

import argparse
import time
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

    cam_state = None
    while True:
        data = mujoco.MjData(model)
        reset(model, data)
        print(f"Loaded: nbody={model.nbody} njnt={model.njnt} nu={model.nu}")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            if cam_state is not None:
                viewer.cam.lookat[:], viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam_state

            files = snapshot()
            last_poll = time.time()
            new_model = None
            while viewer.is_running() and new_model is None:
                step_start = time.time()
                if args.frozen:
                    mujoco.mj_forward(model, data)
                else:
                    mujoco.mj_step(model, data)
                viewer.sync()

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
                return
            cam_state = (viewer.cam.lookat.copy(), viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation)
        model = new_model


if __name__ == "__main__":
    main()
