"""Sanity checks for the generated robot. Prints PASS/FAIL per check.

    python -m tools.check_model

Run after changing the generator or robot.yaml. Exits with code 1 if any check fails.
"""

import sys

import mujoco
import numpy as np

from model.config import load_config
from model.generator import build, build_leg_fixed_base, build_legs_fixed_base

LEG_JOINTS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def sensor(m, d, name):
    s = m.sensor(name)
    return d.sensordata[s.adr[0]:s.adr[0] + s.dim[0]]


def set_joints(m, d, values):
    for joint, value in values.items():
        d.qpos[m.joint(joint).qposadr[0]] = value
    mujoco.mj_forward(m, d)


def settle(m, seconds):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return d


def main() -> None:
    cfg = load_config()

    print("Build")
    for fn in (build, build_leg_fixed_base, build_legs_fixed_base):
        try:
            fn().compile()
            check(f"{fn.__name__} compiles", True)
        except Exception as e:
            check(f"{fn.__name__} compiles", False, str(e).splitlines()[0])
            sys.exit(1)

    m = build().compile()
    d = mujoco.MjData(m)
    mass = m.body_mass.sum()
    check("13 actuated joints", m.nu == 13, f"nu={m.nu}")
    check("mass in 1.2 to 1.5 kg", 1.2 <= mass <= 1.5, f"{mass:.3f} kg")

    print("Joint signs (left leg and head)")
    def knee_point():
        thigh = m.body("left_thigh").id
        return d.xpos[thigh] + d.xmat[thigh].reshape(3, 3) @ [0, 0, -cfg["leg"]["thigh_length"]]
    mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d); knee0 = knee_point()
    set_joints(m, d, {"left_hip_pitch": -0.3})
    check("hip pitch < 0 swings the thigh forward", knee_point()[0] > knee0[0] + 0.01)
    mujoco.mj_resetData(m, d); set_joints(m, d, {"left_hip_roll": 0.3})
    check("hip roll > 0 swings the leg outward", knee_point()[1] > knee0[1] + 0.01)
    mujoco.mj_resetData(m, d); set_joints(m, d, {"left_ankle": -0.3})
    check("ankle < 0 lifts the toes", d.xmat[m.body("left_foot").id].reshape(3, 3)[2, 0] > 0.1)
    mujoco.mj_resetData(m, d); set_joints(m, d, {"neck_pitch": -0.3})
    check("neck pitch < 0 looks up", d.xmat[m.body("head").id].reshape(3, 3)[2, 0] > 0.1)

    print("Mirror symmetry")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        mujoco.mj_resetData(m, d)
        q = {j: rng.uniform(*m.joint(f"left_{j}").range) for j in LEG_JOINTS}
        set_joints(m, d, {f"{side}_{j}": v for j, v in q.items() for side in ("left", "right")})
        for link in ("thigh", "shin", "foot"):
            left, right = d.xpos[m.body(f"left_{link}").id], d.xpos[m.body(f"right_{link}").id]
            worst = max(worst, float(np.abs(left * [1, -1, 1] - right).max()))
    check("right leg mirrors left over 100 random poses", worst < 1e-9, f"max error {worst:.1e} m")

    print("Stand keyframe")
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id); mujoco.mj_forward(m, d)
    soles = [d.geom_xpos[m.geom(f"{s}_foot").id][2] - m.geom_size[m.geom(f"{s}_foot").id][2] for s in ("left", "right")]
    check("feet start on the floor", max(abs(z) for z in soles) < 1e-4, f"sole z = {soles[0]:.5f}, {soles[1]:.5f} m")
    start_z = d.qpos[2]
    d = settle(m, 5.0)
    upright = d.xmat[m.body("torso").id][8]
    check("stands for 5 s", upright > 0.99 and abs(d.qpos[2] - start_z) < 0.005,
          f"torso moved {1000 * (d.qpos[2] - start_z):+.1f} mm, upright {upright:.3f}")
    torque = np.abs(d.actuator_force).max()
    check("standing torque under 50% of stall", torque < 0.5 * cfg["servos"][cfg["default_servo"]]["stall_torque"],
          f"{torque:.3f} N*m")

    print("Sensors")
    weight = mass * 9.81
    feet = sensor(m, d, "left_foot_touch")[0] + sensor(m, d, "right_foot_touch")[0]
    check("foot touch adds up to body weight", abs(feet - weight) < 0.05 * weight, f"{feet:.2f} N vs {weight:.2f} N")
    accel = sensor(m, d, "imu_accel")
    check("accelerometer reads about +9.81 up", abs(accel[2] - 9.81) < 0.3, f"{np.round(accel, 2)}")
    check("gyro reads about 0", np.abs(sensor(m, d, "imu_gyro")).max() < 0.01)
    cam = d.cam_xmat[m.camera("head_camera").id].reshape(3, 3)
    check("head camera looks forward", -cam[0, 2] > 0.95, f"view dir {np.round(-cam[:, 2], 2)}")

    print("Collisions")
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    set_joints(m, d, {"left_hip_roll": -0.5, "right_hip_roll": -0.5})
    names = {m.geom(c.geom1).name + "/" + m.geom(c.geom2).name for c in d.contact[:d.ncon]}
    check("crossed legs collide", any("left_" in n and "right_" in n for n in names), ", ".join(sorted(names)))
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    set_joints(m, d, {"left_knee": 2.27})
    check("same-leg links don't collide", d.ncon == 0, f"{d.ncon} contacts")

    print("Payload")
    payload_cfg = load_config()
    payload_cfg["payload"]["enabled"] = True
    mp = build(payload_cfg).compile()
    added = mp.body_mass.sum() - mass
    check("payload adds its mass", abs(added - cfg["payload"]["mass"]) < 1e-6, f"+{added * 1000:.0f} g")
    dp = settle(mp, 3.0)
    check("stands for 3 s with payload", dp.xmat[mp.body("torso").id][8] > 0.99)

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
