# biped-rl

A from-scratch, ~30 cm, 13-servo bipedal home companion robot. The project
asks how animation and RL techniques from entertainment robotics can make
in-home robots more useful, not just more charming: in a small floor-level
robot, expressive motion and functional motion can be the same motion.

Planned tasks: object memory and search (including under furniture),
come-to-you reminders, drink delivery with a 400 g payload (stretch), and
always-on companionship behaviors.

## Status

Phase 0, software-first design (late Oct to Nov 2026):

- [x] Repo and environment setup, dev tools
- [x] Parametric MuJoCo model generator (MjSpec) driven by one config
- [ ] Keyframed reference motions (walk, look up, attention bob, crouch-peek, point, sit, sleep)
- [ ] Inverse dynamics and actuator sizing sweeps, with and without payload
- [ ] Servo bench tests and BAM actuator model fitting

Later: CAD and one-leg validation, custom PCB and full build, animation-driven
RL in mjlab with sim-to-real, perception and tasks, paper.

## Robot (current guesses, will change after sizing)

- ~30 cm tall, ~1.2 to 1.5 kg
- 13 DOF: per leg hip yaw, hip roll, hip pitch, knee, ankle; neck pitch, neck yaw, jaw
- Feetech STS3215 serial bus servos
- Raspberry Pi 5 (policy at 50 Hz, perception) and an ESP32-S3 carrier board (servo bus, IMU, safety, power)
- Head camera, mic, speaker, touch pad, LED eyes; 2S Li-ion

## Setup

Requires Python 3.11 or 3.12. On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m mujoco.viewer   # check that the viewer opens
```

If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Dev tools

Run from the repo root with the venv active.

```powershell
python -m tools.view                       # hot-reload viewer for model.generator:build
python -m tools.view model.leg:build_test  # any module:function returning an MjSpec or MjModel
python -m tools.view path\to\robot.xml     # or a plain MJCF file
python -m tools.view --frozen              # kinematics only, sliders set joint angles

python -m tools.inspect_model              # mass, COM, joints, actuators, sensors
python -m tools.check_model                # PASS/FAIL sanity checks (signs, mirror, standing, sensors)
```

The viewer rebuilds whenever a file under `model/` is saved and keeps the
last good model if the rebuild fails. A slider window opens next to it: with
physics on, sliders set the servo targets; with `--frozen`, they set joint
angles directly. Built models are written to
`model/generated/robot.xml` (not committed).

## Conventions

- Frames: x forward, y left, z up. SI units everywhere (m, kg, N*m, rad).
- Left and right legs are mirrored.
- All tunable robot parameters live in one config.

## Repo layout

```
tools/        dev tools: hot-reload viewer, model inspector
model/        parametric MJCF generator and config
motions/      keyframe files, playback and editing tools
analysis/     inverse dynamics, sizing sweeps, plots
actuator_id/  servo bench logger, BAM fitting
training/     mjlab tasks and configs
firmware/     ESP32-S3 firmware
runtime/      Pi: policy runner, behaviors, perception
paper/        LaTeX paper
```

## References

- Grandia et al., "Design and Control of a Bipedal Robotic Character" (Disney, arXiv 2501.05204)
- Duclusaud et al., "Extended Friction Models for the Physics Simulation of Servo Actuators" (BAM, arXiv 2410.08650)
- [Microduck](https://github.com/pollen-robotics/microduck), Open Duck Mini, XGO-Duck (software and proportion references only)
