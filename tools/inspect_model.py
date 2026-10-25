"""Print a summary of the model: mass, center of mass, joints, actuators, sensors.

    python -m tools.inspect_model                  # uses model.generator:build
    python -m tools.inspect_model path/to/robot.xml

Center of mass is computed at the "stand" keyframe if present, else at qpos0.
"""

import argparse

import mujoco
import numpy as np

from tools.loader import DEFAULT_SOURCE, load

JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def name(model, obj, i):
    return mujoco.mj_id2name(model, obj, i) or f"<{i}>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default=DEFAULT_SOURCE)
    args = parser.parse_args()

    model = load(args.source)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    # Body 0 is the world; everything else counts toward the robot.
    total_mass = model.body_mass[1:].sum()
    com = (model.body_mass[1:, None] * data.xipos[1:]).sum(axis=0) / total_mass
    pose = "stand keyframe" if key >= 0 else "qpos0"

    print(f"Total mass: {total_mass * 1000:.1f} g")
    print(f"COM ({pose}): x={com[0]*100:+.2f} y={com[1]*100:+.2f} z={com[2]*100:+.2f} cm")
    print(f"timestep={model.opt.timestep}  nq={model.nq} nv={model.nv} nu={model.nu}")

    print(f"\nBodies ({model.nbody - 1}):")
    for i in range(1, model.nbody):
        print(f"  {name(model, mujoco.mjtObj.mjOBJ_BODY, i):<20} {model.body_mass[i]*1000:8.1f} g"
              f"  parent={name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[i])}")

    print(f"\nJoints ({model.njnt}):")
    for i in range(model.njnt):
        jtype = JOINT_TYPES[model.jnt_type[i]]
        line = f"  {name(model, mujoco.mjtObj.mjOBJ_JOINT, i):<20} {jtype:<6}"
        if jtype in ("hinge", "slide"):
            axis = np.round(model.jnt_axis[i], 3)
            lo, hi = np.degrees(model.jnt_range[i]) if jtype == "hinge" else model.jnt_range[i]
            limited = "" if model.jnt_limited[i] else " (unlimited)"
            dof = model.jnt_dofadr[i]
            line += (f" axis={axis}  range=[{lo:+.0f}, {hi:+.0f}]{limited}"
                     f"  damping={model.dof_damping[dof]:.3g} armature={model.dof_armature[dof]:.3g}")
        print(line)

    print(f"\nActuators ({model.nu}):")
    for i in range(model.nu):
        target = name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
        kp = model.actuator_gainprm[i, 0]
        frc = model.actuator_forcerange[i] if model.actuator_forcelimited[i] else None
        frc_str = f"[{frc[0]:+.2f}, {frc[1]:+.2f}] Nm" if frc is not None else "unlimited"
        print(f"  {name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):<20} -> {target:<20} kp={kp:<8.3g} force={frc_str}")

    print(f"\nSensors ({model.nsensor}):")
    for i in range(model.nsensor):
        stype = mujoco.mjtSensor(model.sensor_type[i]).name.removeprefix("mjSENS_").lower()
        print(f"  {name(model, mujoco.mjtObj.mjOBJ_SENSOR, i):<20} {stype}  dim={model.sensor_dim[i]}")

    print(f"\nCameras ({model.ncam}): " + ", ".join(name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)))
    print(f"Sites ({model.nsite}): " + ", ".join(name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)))
    print(f"Keyframes ({model.nkey}): " + ", ".join(name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)))


if __name__ == "__main__":
    main()
