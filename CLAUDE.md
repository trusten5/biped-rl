# CLAUDE.md

Project brief for Claude Code. Read this at the start of every session.

## What this project is

A from-scratch, ~30 cm, 13-servo bipedal home companion robot, built by
Trusten (Harvey Mudd engineering student) as a research project with a
paper-style write-up as a deliverable.

**Research thesis:** How can animation and RL techniques from entertainment
robotics (Disney's bipedal character, Pollen Robotics' Microduck) make
in-home robots more useful, not just more charming?

**Core idea:** in a small floor-level robot, expressive motion and
functional motion can be the same motion. Peeking under the couch is how
the camera searches; walking over and looking up at you is how a reminder
gets noticed.

**Tasks:**
1. Object memory and search: remembers where it last saw items (keys,
   AirPods, remote), searches including under furniture, points to them.
2. Come-to-you reminders: finds the person, approaches, looks up, attention
   bob, spoken reminder, head-pat to acknowledge, patient re-nudge.
3. Stretch: drink delivery with a body-mounted holder (400 g payload).
4. Always on: companionship behaviors (sit, idle, sleep).

## Current phase

**Phase 0 (late Oct to Nov 2026): software-first design.**
1. Parametric MuJoCo model generator (MjSpec, Python) built from a config.
2. Keyframed reference motions on that model (walk, look up, attention bob,
   crouch-peek, point, patient wait, sit, sleep).
3. Inverse dynamics on each motion, with and without a 400 g payload,
   sweeping leg length and mass until joint torques fit the servos with
   margin.
4. When servos arrive: BAM-style pendulum bench, fit actuator models, rerun
   sizing with measured limits.

Later phases: CAD and one-leg validation (Dec to Jan), custom PCB and full
build (Jan to Feb), animation-driven RL in mjlab with sim-to-real (Feb to
Apr), perception, tasks and paper (Apr to May).

**Status (2026-10-25):** Step 1 (repo setup) done. Next: parameter config,
then one leg on a fixed base (build_leg), then the full robot.

## Working with Claude

- Trusten writes the core code (config, generator, motions, analysis).
  Claude tutors, explains, and reviews; it does not write core code unless
  asked.
- Claude may write boilerplate and dev tooling (tools/, configs, scripts).
- Reviews: check structure, joint axes and signs, mirroring, units, and
  anything that will cause trouble for inverse dynamics or RL later.

## Robot spec (current guesses, will change after sizing)

- ~30 cm tall, target mass ~1.2 to 1.5 kg
- 13 DOF: per leg hip yaw, hip roll, hip pitch, knee, ankle (5 x 2), neck
  pitch, neck yaw, jaw
- Servos: Feetech STS3215 serial bus servos (Feetech 1910 as a smaller
  alternative); compare after bench tests
- Compute: Raspberry Pi 5 (policy at 50 Hz, perception), ESP32-S3 on a
  custom carrier board (servo bus, ICM-42688 IMU, safety, power)
- Sensors: head camera, mic, speaker, capacitive touch pad on head, LED eyes
  that double as a headlamp
- Power: 2S Li-ion with BMS
- Payload holder near the hips for the drink stretch task

## Tech stack

- Python 3.11 or 3.12, MuJoCo (MjSpec API for model building)
- mjlab (MuJoCo Warp) for GPU RL training later; PPO
- BAM (Better Actuator Models, Rhoban) for actuator identification
- Onshape + onshape-to-robot for CAD to MJCF later
- PlatformIO / ESP-IDF for ESP32-S3 firmware (C/C++)
- Paper in LaTeX (IEEEtran conference format) under paper/

## Repo layout

```
tools/        dev tools: hot-reload viewer, model inspector, shared loader
model/        parametric MJCF generator, config, generated models (gitignored)
motions/      keyframe files, playback and editing tools
analysis/     inverse dynamics, sizing sweeps, plots for the paper
actuator_id/  servo bench logger, BAM fitting
training/     mjlab tasks and configs (Phase 3)
firmware/     ESP32-S3 (Phase 2)
runtime/      Pi: policy runner, behaviors, perception, memory, scheduler
paper/        LaTeX paper
```

## Conventions

- Frames: x forward, y left, z up. SI units everywhere (m, kg, N*m, rad).
- Left and right legs are mirrored; check sign flips on hip yaw and roll.
- All tunable robot parameters live in one config; no magic numbers in the
  generator.
- Generated files go in model/generated/ and are not committed.
- Every analysis script saves its figures to a file (they feed the paper).

## Environment

- Development on Windows 11 with an NVIDIA RTX GPU, PowerShell terminal.
- GPU training (mjlab) may need WSL2; check before Phase 3.
- Repo lives at C:\Projects\biped-rl. Keep it out of OneDrive-synced folders.
- Python 3.12 venv in .venv/. Activate with `.\.venv\Scripts\Activate.ps1`.

## Dev tools

Run from the repo root with the venv active.

- `python -m tools.view [module:func | file.xml] [--frozen]`: hot-reload
  viewer. Rebuilds whenever a file under model/ is saved; keeps the last good
  model on errors; resets to the "stand" keyframe if present.
- `python -m tools.inspect_model [module:func | file.xml]`: prints total
  mass, COM, bodies, joints, actuators, sensors, cameras, sites, keyframes.
- Default source is `model.generator:build`: a zero-argument function that
  loads the config and returns an MjSpec. When built from a spec, the XML is
  written to model/generated/robot.xml for debugging.

## Key references

- Grandia et al., "Design and Control of a Bipedal Robotic Character"
  (Disney, arXiv 2501.05204): motion-first design, animation-tracking RL
- Duclusaud et al., "Extended Friction Models for the Physics Simulation of
  Servo Actuators" (BAM, arXiv 2410.08650)
- Microduck (github.com/pollen-robotics/microduck), Open Duck Mini, XGO-Duck
  (github.com/LuwuDynamics/xgoduck_hardware) as software and proportion
  references only; do not copy their hardware designs.

## Writing style

When writing docs, comments meant for the paper, or commit messages: plain
language, no em dashes.