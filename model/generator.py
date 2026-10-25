"""Parametric MjSpec generator. Builds the robot from robot.yaml.

Outline (to implement):
  1. build_leg(spec, parent, cfg, side): hip yaw -> hip roll -> hip pitch ->
     thigh -> knee -> shin -> ankle -> foot. side = +1 (left) or -1 (right);
     multiply y offsets by side and flip hip yaw and roll axis signs for the right.
  2. Add servo mass and inertia to the link that holds each servo.
  3. Add a position actuator per joint from the servo params.
  4. Add sensors (IMU site on torso, foot contact), cameras, and keyframes.
"""

import mujoco

from model.config import load_config


def build_leg_fixed_base() -> mujoco.MjSpec:
    """One left leg hanging from a fixed base, for checking axes and signs.

    View with: python -m tools.view model.generator:build_leg_fixed_base
    """
    cfg = load_config()
    leg = cfg['leg']
    joints = cfg['joints']

    spec = mujoco.MjSpec()
    spec.compiler.degree = False  # all angles in radians
    spec.option.timestep = cfg['sim']['timestep']

    # Scene
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0.5, 0.5, 0.01], rgba=[0.5, 0, 0.5, 1])
    spec.worldbody.add_light(name='light', pos=[0, 0, 1], dir=[0, 0, -1])

    # Fixed mount standing in for the torso (no joint, so welded to the world)
    mount = spec.worldbody.add_body(name='mount', pos=[0, 0, 0.3])
    mount.add_geom(name='mount', type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.02, 0.02, 0.02], rgba=[0, 1, 0, 1])

    # Hip yaw link: body frame sits on the hip yaw axis
    hip_yaw_link = mount.add_body(name='left_hip_yaw_link', pos=leg['hip_offset'])
    hip_yaw_link.add_joint(name='left_hip_yaw', type=mujoco.mjtJoint.mjJNT_HINGE, axis=joints['hip_yaw']['axis'], range=joints['hip_yaw']['range'])
    hip_yaw_link.add_geom(name='left_hip_yaw_link', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.008, 0.01], pos=[0, 0, -0.01], mass=leg['structure_mass']['hip_yaw_link'])

    # Hip roll link: body frame sits on the hip roll axis (long in y so the roll is visible)
    hip_roll_link = hip_yaw_link.add_body(name='left_hip_roll_link', pos=leg['hip_yaw_to_roll'])
    hip_roll_link.add_joint(name='left_hip_roll', type=mujoco.mjtJoint.mjJNT_HINGE, axis=joints['hip_roll']['axis'], range=joints['hip_roll']['range'])
    hip_roll_link.add_geom(name='left_hip_roll_link', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.008, 0.015, 0.0075], pos=[0, 0, -0.0075], mass=leg['structure_mass']['hip_roll_link'])

    # Thigh: body frame sits on the hip pitch axis, capsule runs down to the knee
    thigh_length = leg['thigh_length']
    thigh = hip_roll_link.add_body(name='left_thigh', pos=leg['hip_roll_to_pitch'])
    thigh.add_joint(name='left_hip_pitch', type=mujoco.mjtJoint.mjJNT_HINGE, axis=joints['hip_pitch']['axis'], range=joints['hip_pitch']['range'])
    thigh.add_geom(name='left_thigh', type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.012], fromto=[0, 0, 0, 0, 0, -thigh_length], mass=leg['structure_mass']['thigh'])

    # Shin: body frame sits on the knee axis, capsule runs down to the ankle
    shin_length = leg['shin_length']
    shin = thigh.add_body(name='left_shin', pos=[0, 0, -thigh_length])
    shin.add_joint(name='left_knee', type=mujoco.mjtJoint.mjJNT_HINGE, axis=joints['knee']['axis'], range=joints['knee']['range'])
    shin.add_geom(name='left_shin', type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.011], fromto=[0, 0, 0, 0, 0, -shin_length], mass=leg['structure_mass']['shin'])

    # Foot: body frame sits on the ankle axis; YAML gives full box size, MuJoCo wants half sizes
    foot_cfg = leg['foot']
    foot = shin.add_body(name='left_foot', pos=[0, 0, -shin_length])
    foot.add_joint(name='left_ankle', type=mujoco.mjtJoint.mjJNT_HINGE, axis=joints['ankle']['axis'], range=joints['ankle']['range'])
    foot.add_geom(name='left_foot', type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in foot_cfg['size']], pos=foot_cfg['ankle_to_center'], mass=leg['structure_mass']['foot'])

    return spec


def build() -> mujoco.MjSpec:
    """Full robot with a free-floating torso. Default source for the dev tools."""
    cfg = load_config()
    raise NotImplementedError
