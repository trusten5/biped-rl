"""Parametric MjSpec generator. Builds the robot from robot.yaml.

Outline:
  1. build_leg(spec, parent, cfg, side): hip yaw -> hip roll -> hip pitch ->
     thigh -> knee -> shin -> ankle -> foot. side = +1 (left) or -1 (right). Done.
  2. Servo mass on the link that holds each servo, and a position actuator per
     joint (add_servo_joint). Done.
  3. Stand keyframe with automatic base height (add_keyframes). Done.
  4. Sensors (torso IMU, foot and head touch), head camera, and payload. Done.
  5. Collision filtering (set_collisions). Done.

Sensors mirror the real hardware where they can: the ICM-42688 IMU on the
carrier board (gyro, accelerometer), foot contact, the capacitive head pad,
and the head camera. imu_quat is sim only (on the robot it comes from IMU
fusion on the ESP32).

Servos: each joint is driven by one servo. The servo body is bolted to the
parent link and its horn turns the child, so the servo's mass goes in the
parent body at the joint axis. A joint uses default_servo unless its YAML
entry has a servo key.

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


# Collision bits (contype / conaffinity). Two geoms collide if
# (contype1 & conaffinity2) or (contype2 & conaffinity1) is nonzero.
WORLD = 1
ROBOT = 2


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
                            size=[0.5, 0.5, 0.01], rgba=[0.5, 0, 0.5, 1], contype=WORLD, conaffinity=ROBOT)
    spec.worldbody.add_light(name='light', pos=[0, 0, 1], dir=[0, 0, -1])
    return spec


def add_servo_joint(spec, parent, body, cfg, joint, name, axis):
    """Add a hinge driven by a servo: the joint on body, the servo's mass in parent, and a position actuator.

    joint is the key in cfg['joints']; name is the full name (for example left_knee).
    """
    joint_cfg = cfg['joints'][joint]
    servo = cfg['servos'][joint_cfg.get('servo', cfg['default_servo'])]

    # Joint, with the servo's rotor inertia and friction
    body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis, range=joint_cfg['range'], armature=servo['armature'], damping=servo['damping'], frictionloss=servo['frictionloss'])

    # Servo case in the parent, centered on the joint axis (mass only, no collisions)
    parent.add_geom(name=f'{name}_servo', type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in servo['size']], pos=list(body.pos), mass=servo['mass'], contype=0, conaffinity=0, rgba=[0.15, 0.15, 0.15, 1])

    # Position actuator: ctrl is the target angle, torque = kp * (ctrl - angle), clipped to stall torque
    actuator = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    actuator.set_to_position(kp=servo['kp'], inheritrange=True)
    actuator.forcerange = [-servo['stall_torque'], servo['stall_torque']]


def build_leg(spec, parent, cfg, side):
    """Add one leg under parent. side = +1 for left, -1 for right. Returns the foot body."""
    leg = cfg['leg']
    joints = cfg['joints']
    prefix = 'left' if side > 0 else 'right'
    mass = leg['structure_mass']

    def add_hinge(parent, body, joint):
        add_servo_joint(spec, parent, body, cfg, joint, f'{prefix}_{joint}', mirror_axis(joints[joint]['axis'], side))

    # Hip yaw link: body frame sits on the hip yaw axis
    hip_yaw_link = parent.add_body(name=f'{prefix}_hip_yaw_link', pos=mirror_pos(leg['hip_offset'], side))
    add_hinge(parent, hip_yaw_link, 'hip_yaw')
    hip_yaw_link.add_geom(name=f'{prefix}_hip_yaw_link', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.008, 0.01], pos=[0, 0, -0.01], mass=mass['hip_yaw_link'])

    # Hip roll link: body frame sits on the hip roll axis (long in y so the roll is visible)
    hip_roll_link = hip_yaw_link.add_body(name=f'{prefix}_hip_roll_link', pos=mirror_pos(leg['hip_yaw_to_roll'], side))
    add_hinge(hip_yaw_link, hip_roll_link, 'hip_roll')
    hip_roll_link.add_geom(name=f'{prefix}_hip_roll_link', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.008, 0.015, 0.0075], pos=[0, 0, -0.0075], mass=mass['hip_roll_link'])

    # Thigh: body frame sits on the hip pitch axis, capsule runs down to the knee
    thigh_length = leg['thigh_length']
    thigh = hip_roll_link.add_body(name=f'{prefix}_thigh', pos=mirror_pos(leg['hip_roll_to_pitch'], side))
    add_hinge(hip_roll_link, thigh, 'hip_pitch')
    thigh.add_geom(name=f'{prefix}_thigh', type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.012], fromto=[0, 0, 0, 0, 0, -thigh_length], mass=mass['thigh'])

    # Shin: body frame sits on the knee axis, capsule runs down to the ankle
    shin_length = leg['shin_length']
    shin = thigh.add_body(name=f'{prefix}_shin', pos=[0, 0, -thigh_length])
    add_hinge(thigh, shin, 'knee')
    shin.add_geom(name=f'{prefix}_shin', type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.011], fromto=[0, 0, 0, 0, 0, -shin_length], mass=mass['shin'])

    # Foot: body frame sits on the ankle axis; YAML gives full box size, MuJoCo wants half sizes
    foot_cfg = leg['foot']
    foot = shin.add_body(name=f'{prefix}_foot', pos=[0, 0, -shin_length])
    add_hinge(shin, foot, 'ankle')
    foot.add_geom(name=f'{prefix}_foot', type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in foot_cfg['size']], pos=mirror_pos(foot_cfg['ankle_to_center'], side), mass=mass['foot'])

    # Foot touch sensor: a thin site volume covering the sole; it reports the normal force of contacts inside it.
    # Box contacts sit on the sole's corners, so the site is a little larger than the sole to keep them inside.
    margin = 0.003
    sole_center = mirror_pos(foot_cfg['ankle_to_center'], side)
    sole_center[2] -= foot_cfg['size'][2] / 2
    foot.add_site(name=f'{prefix}_foot_touch', type=mujoco.mjtGeom.mjGEOM_BOX, size=[foot_cfg['size'][0] / 2 + margin, foot_cfg['size'][1] / 2 + margin, margin], pos=sole_center, rgba=[1, 0.5, 0, 0.3])
    spec.add_sensor(name=f'{prefix}_foot_touch', type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f'{prefix}_foot_touch')

    return foot


def build_head(spec, parent, cfg):
    """Add the pan-tilt neck, head, and jaw under parent. On the centerline, so not mirrored. Returns the head body."""
    head_cfg = cfg['head']
    joints = cfg['joints']

    def add_hinge(parent, body, joint):
        add_servo_joint(spec, parent, body, cfg, joint, joint, joints[joint]['axis'])

    # Neck link: body frame sits on the neck yaw axis, small bracket up to the pitch axis
    neck_link = parent.add_body(name='neck_link', pos=head_cfg['neck_offset'])
    add_hinge(parent, neck_link, 'neck_yaw')
    neck_link.add_geom(name='neck_link', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.01, 0.005], pos=[0, 0, 0.0075], mass=head_cfg['neck_link_mass'])

    # Head: body frame sits on the neck pitch axis, box sits on top of it
    head = neck_link.add_body(name='head', pos=head_cfg['neck_yaw_to_pitch'])
    add_hinge(neck_link, head, 'neck_pitch')
    head.add_geom(name='head', type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in head_cfg['size']], pos=head_cfg['center'], mass=head_cfg['structure_mass'])

    # Jaw: body frame sits on the jaw hinge at the back of the mouth, flap extends forward
    jaw = head.add_body(name='jaw', pos=head_cfg['jaw_hinge'])
    add_hinge(head, jaw, 'jaw')
    jaw.add_geom(name='jaw', type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in head_cfg['jaw_size']], pos=head_cfg['jaw_center'], mass=head_cfg['jaw_mass'])

    # Head camera looking forward along the head's +x. A MuJoCo camera looks along its own -z,
    # so xyaxes sets camera x = head -y (image right) and camera y = head +z (image up).
    head.add_camera(name='head_camera', pos=head_cfg['camera_pos'], xyaxes=[0, -1, 0, 0, 0, 1], fovy=head_cfg['camera_fovy'])

    # Capacitive touch pad on top of the head, for head pats
    head.add_site(name='head_touch', type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in head_cfg['touch_pad_size']], pos=head_cfg['touch_pad_pos'], rgba=[0, 0.6, 1, 0.3])
    spec.add_sensor(name='head_touch', type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname='head_touch')

    return head


def add_imu(spec, torso, cfg):
    """IMU site on the torso with gyro, accelerometer, and orientation sensors."""
    torso.add_site(name='imu', pos=cfg['torso']['imu_pos'], size=[0.005])
    site = dict(objtype=mujoco.mjtObj.mjOBJ_SITE, objname='imu')
    spec.add_sensor(name='imu_gyro', type=mujoco.mjtSensor.mjSENS_GYRO, **site)
    spec.add_sensor(name='imu_accel', type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, **site)
    spec.add_sensor(name='imu_quat', type=mujoco.mjtSensor.mjSENS_FRAMEQUAT, **site)


def add_payload(torso, cfg):
    """Drink payload fixed to the torso, if enabled. Mass only for now (no collisions)."""
    payload = cfg['payload']
    if not payload['enabled']:
        return
    body = torso.add_body(name='payload', pos=payload['position'])
    diameter, height = payload['size']
    body.add_geom(name='payload', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[diameter / 2, height / 2], mass=payload['mass'], contype=0, conaffinity=0, rgba=[0.9, 0.7, 0.1, 1])


def set_collisions(spec, cfg):
    """Robot geoms collide with the floor only, plus explicit left vs right leg pairs from the YAML."""
    for geom in spec.geoms:
        if geom.name == 'floor' or geom.contype == 0:
            continue  # leave the floor and mass-only geoms (servos, payload) alone
        geom.contype, geom.conaffinity = ROBOT, WORLD

    # Explicit pairs bypass the contype / conaffinity filter
    for a, b in cfg['collisions']['leg_pairs']:
        spec.add_pair(name=f'left_{a}_right_{b}', geomname1=f'left_{a}', geomname2=f'right_{b}')
        if a != b:
            spec.add_pair(name=f'right_{a}_left_{b}', geomname1=f'right_{a}', geomname2=f'left_{b}')


LEG_JOINTS = ['hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle']


def add_keyframes(spec, cfg):
    """Add each pose in cfg['keyframes'] as a MuJoCo keyframe.

    Leg joints in the YAML (hip_pitch, knee, ...) are set on both legs; other
    keys are full joint names. ctrl is set equal to the pose so the position
    actuators hold it instead of pulling to zero. base_height 0 means: place
    the torso so the lowest point of the robot just touches the floor.
    """
    model = spec.compile()
    data = mujoco.MjData(model)
    root = model.joint('root').qposadr[0]

    for key_name, pose in cfg['keyframes'].items():
        data.qpos[:] = model.qpos0
        for joint, value in pose.items():
            if joint == 'base_height':
                continue
            names = [f'left_{joint}', f'right_{joint}'] if joint in LEG_JOINTS else [joint]
            for name in names:
                data.qpos[model.joint(name).qposadr[0]] = value

        data.qpos[root:root + 3] = [0, 0, 0]
        height = pose.get('base_height', 0) or -_lowest_point(model, data)
        data.qpos[root + 2] = height

        # Each actuator's target is the angle of the joint it drives
        ctrl = [data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]] for i in range(model.nu)]
        spec.add_key(name=key_name, qpos=data.qpos.copy(), ctrl=ctrl)


def _lowest_point(model, data):
    """Lowest world z of any colliding robot geom (boxes, capsules, spheres) at data.qpos."""
    mujoco.mj_forward(model, data)
    lowest = float('inf')
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 or model.geom_contype[g] == 0:
            continue  # skip the floor and mass-only servo boxes
        center, rot, size = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        gtype = model.geom_type[g]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            # z of the lowest corner: center minus the half sizes projected onto world z
            z = center[2] - sum(abs(rot[2, i]) * size[i] for i in range(3))
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            z = center[2] - abs(rot[2, 2]) * size[1] - size[0]
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = center[2] - size[0]
        else:
            continue
        lowest = min(lowest, z)
    return lowest


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
    build_leg(spec, _fixed_mount(spec), cfg, side)
    return spec


def build_legs_fixed_base() -> mujoco.MjSpec:
    """Both legs on a fixed base, for checking that the right leg mirrors the left.

    View with: python -m tools.view model.generator:build_legs_fixed_base --frozen
    """
    cfg = load_config()
    spec = new_spec(cfg)
    mount = _fixed_mount(spec)
    build_leg(spec, mount, cfg, 1)
    build_leg(spec, mount, cfg, -1)
    return spec


def build(cfg=None) -> mujoco.MjSpec:
    """Full robot with a free-floating torso. Default source for the dev tools.

    Pass a cfg dict to build a variant (sizing sweeps, payload on); otherwise robot.yaml is loaded.
    """
    cfg = cfg or load_config()
    spec = new_spec(cfg)

    # Torso (free joint, so the whole robot can move and fall)
    torso = spec.worldbody.add_body(name='torso', pos=[0, 0, 0.3])
    torso.add_freejoint(name='root')
    torso.add_geom(name='torso', type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in cfg['torso']['size']], mass = cfg['torso']['structure_mass'], rgba=[0, 1, 0, 1])

    # use leg helpers
    build_leg(spec, torso, cfg, 1)
    build_leg(spec, torso, cfg, -1)
    build_head(spec, torso, cfg)
    add_imu(spec, torso, cfg)
    add_payload(torso, cfg)
    set_collisions(spec, cfg)

    add_keyframes(spec, cfg)
    return spec
