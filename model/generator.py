"""Parametric MjSpec generator. Builds the robot from robot.yaml.

Outline:
  1. build_leg(parent, cfg, side): hip yaw -> hip roll -> hip pitch ->
     thigh -> knee -> shin -> ankle -> foot. side = +1 (left) or -1 (right). Done.
  2. Add servo mass and inertia to the link that holds each servo.
  3. Add a position actuator per joint from the servo params.
  4. Add sensors (IMU site on torso, foot contact), cameras, and keyframes.

Mirroring: the YAML describes the left leg. For the right leg (side = -1),
positions flip their y component, and joint axes flip their x and z
components. Axes flip differently because a rotation axis is not an ordinary
vector: reflecting the robot across the xz plane turns a +x axis into -x.
With this rule every joint means the same thing on both sides (positive hip
roll swings either leg outward, positive hip yaw turns either foot outward),
and pitch axes (along y) are unchanged.
"""

import mujoco

from model.config import load_config


def mirror_pos(v, side):
    """Mirror a position across the xz plane for the right side."""
    return [v[0], side * v[1], v[2]]


def mirror_axis(a, side):
    """Mirror a joint axis across the xz plane for the right side."""
    return [side * a[0], a[1], side * a[2]]


def new_spec(cfg) -> mujoco.MjSpec:
    """Empty spec with compiler settings, floor, and light."""
    spec = mujoco.MjSpec()
    spec.compiler.degree = False  # all angles in radians
    spec.option.timestep = cfg['sim']['timestep']

    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0.5, 0.5, 0.01], rgba=[0.5, 0, 0.5, 1])
    spec.worldbody.add_light(name='light', pos=[0, 0, 1], dir=[0, 0, -1])
    return spec


def build_leg(parent, cfg, side):
    """Add one leg under parent. side = +1 for left, -1 for right. Returns the foot body."""
    leg = cfg['leg']
    joints = cfg['joints']
    prefix = 'left' if side > 0 else 'right'
    mass = leg['structure_mass']

    def add_hinge(body, joint):
        body.add_joint(name=f'{prefix}_{joint}', type=mujoco.mjtJoint.mjJNT_HINGE, axis=mirror_axis(joints[joint]['axis'], side), range=joints[joint]['range'])

    # Hip yaw link: body frame sits on the hip yaw axis
    hip_yaw_link = parent.add_body(name=f'{prefix}_hip_yaw_link', pos=mirror_pos(leg['hip_offset'], side))
    add_hinge(hip_yaw_link, 'hip_yaw')
    hip_yaw_link.add_geom(name=f'{prefix}_hip_yaw_link', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.008, 0.01], pos=[0, 0, -0.01], mass=mass['hip_yaw_link'])

    # Hip roll link: body frame sits on the hip roll axis (long in y so the roll is visible)
    hip_roll_link = hip_yaw_link.add_body(name=f'{prefix}_hip_roll_link', pos=mirror_pos(leg['hip_yaw_to_roll'], side))
    add_hinge(hip_roll_link, 'hip_roll')
    hip_roll_link.add_geom(name=f'{prefix}_hip_roll_link', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.008, 0.015, 0.0075], pos=[0, 0, -0.0075], mass=mass['hip_roll_link'])

    # Thigh: body frame sits on the hip pitch axis, capsule runs down to the knee
    thigh_length = leg['thigh_length']
    thigh = hip_roll_link.add_body(name=f'{prefix}_thigh', pos=mirror_pos(leg['hip_roll_to_pitch'], side))
    add_hinge(thigh, 'hip_pitch')
    thigh.add_geom(name=f'{prefix}_thigh', type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.012], fromto=[0, 0, 0, 0, 0, -thigh_length], mass=mass['thigh'])

    # Shin: body frame sits on the knee axis, capsule runs down to the ankle
    shin_length = leg['shin_length']
    shin = thigh.add_body(name=f'{prefix}_shin', pos=[0, 0, -thigh_length])
    add_hinge(shin, 'knee')
    shin.add_geom(name=f'{prefix}_shin', type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.011], fromto=[0, 0, 0, 0, 0, -shin_length], mass=mass['shin'])

    # Foot: body frame sits on the ankle axis; YAML gives full box size, MuJoCo wants half sizes
    foot_cfg = leg['foot']
    foot = shin.add_body(name=f'{prefix}_foot', pos=[0, 0, -shin_length])
    add_hinge(foot, 'ankle')
    foot.add_geom(name=f'{prefix}_foot', type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in foot_cfg['size']], pos=mirror_pos(foot_cfg['ankle_to_center'], side), mass=mass['foot'])

    return foot


def _fixed_mount(spec):
    """Fixed box standing in for the torso (no joint, so welded to the world)."""
    mount = spec.worldbody.add_body(name='mount', pos=[0, 0, 0.3])
    mount.add_geom(name='mount', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], rgba=[0, 1, 0, 1])
    return mount


def build_leg_fixed_base(side=1) -> mujoco.MjSpec:
    """One leg hanging from a fixed base, for checking axes and signs.

    View with: python -m tools.view model.generator:build_leg_fixed_base --frozen
    """
    cfg = load_config()
    spec = new_spec(cfg)
    build_leg(_fixed_mount(spec), cfg, side)
    return spec


def build_legs_fixed_base() -> mujoco.MjSpec:
    """Both legs on a fixed base, for checking that the right leg mirrors the left.

    View with: python -m tools.view model.generator:build_legs_fixed_base --frozen
    """
    cfg = load_config()
    spec = new_spec(cfg)
    mount = _fixed_mount(spec)
    build_leg(mount, cfg, 1)
    build_leg(mount, cfg, -1)
    return spec


def build() -> mujoco.MjSpec:
    """Full robot with a free-floating torso. Default source for the dev tools."""
    cfg = load_config()
    raise NotImplementedError
