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
    raise NotImplementedError


def build() -> mujoco.MjSpec:
    """Full robot with a free-floating torso. Default source for the dev tools."""
    cfg = load_config()
    raise NotImplementedError
