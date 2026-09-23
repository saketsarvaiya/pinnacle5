# Pinnacle 5 — Autonomous Delivery Vehicle Prototype

Software proof-of-concept for the perception, route, mission-control, motion, communications, and safety layers of a small autonomous delivery vehicle.

> **Prototype scope:** the accompanying project guide defines a vehicle that carries a 500 g payload, follows a marked indoor route, chooses junctions, docks, verifies unloading, and returns to base. Its proposed architecture separates a host computer from a local ESP32 motion/safety controller.

## What this repository demonstrates

- OpenCV route/tape perception and ArUco marker detection.
- Lightweight ML route-confidence classification.
- A small CNN experiment for route-state classification.
- Differential-drive kinematics and closed-loop simulation.
- A finite-state mission controller for loading, routing, docking, release, return, obstacle waits, and faults.
- A binary host/controller protocol with CRC16, sequence numbers, session IDs, timestamps, and motion limits.
- A mock ESP32 controller with command expiry and local obstacle override.
- Failure-injection tests for the communications/safety layer.

## Architecture

```text
Camera
  |
  v
HOST COMPUTER
  |-- route_vision / route_vision2
  |     |-- ArUco markers
  |     |-- route/tape extraction
  |     `-- route confidence
  |
  |-- mission_fsm
  |     `-- mission state + release interlocks
  |
  `-- host_link_manager + protocol
          |
          | motion commands / telemetry
          v
ESP32 CONTROLLER
  |-- command freshness/watchdog
  |-- obstacle stop
  |-- encoder/sensor interface
  |-- cargo/lid/lock state
  `-- motor driver interface
```

The guide describes the same host/ESP32 separation and explicitly puts local stop behaviour in the controller rather than trusting the host alone.

---

# Scripts

## `route_vision.py` — baseline perception pipeline

Reads a webcam or video source and performs:

1. ArUco detection using `DICT_4X4_50`.
2. Bottom-half floor ROI extraction.
3. Adaptive thresholding for dark route tape.
4. Largest-contour selection.
5. Route centroid calculation.
6. Normalized lateral-error calculation.
7. Lightweight logistic-regression classification.

The classifier predicts:

```text
0 = Valid Route
1 = Uncertain
2 = Junction Candidate
```

Its four features are tape area, recent centroid variance, image brightness, and a simple classical-confidence value.

**Important:** the classifier is trained from a tiny hard-coded synthetic dataset. It is a baseline prototype, not a validated perception model.

Run:

```bash
python route_vision.py
```

Default camera is source `0`. A video filename can be passed to `VisionPipeline(source=...)`.

## `route_vision2.py` — improved perception prototype

Second iteration of the vision pipeline.

Adds:

- Gaussian blur to suppress floor texture.
- More conservative contour filtering.
- `cv2.fitLine()` for route-heading estimation.
- ArUco pose estimation with `cv2.solvePnP()`.
- A threshold-mask debug window.

The code currently uses **placeholder camera calibration** (`fx=600`, `fy=600`, principal point `(320,240)`, zero distortion) and a 5 cm marker. Replace these with measured calibration before using pose for physical docking.

Run:

```bash
python route_vision2.py
```

## `cnn.py` — neural route-state experiment

A separate ML experiment from the OpenCV/logistic-regression pipeline.

### Dataset loader

`load_donkeycar_tub()` recursively searches DonkeyCar `.catalog` files, loads grayscale images, and maps driving metadata to project classes:

```text
throttle < 0.1       -> Uncertain
abs(steering) > 0.6  -> Junction Candidate
otherwise            -> Valid Route
```

### Synthetic data

`generate_synthetic_track_data()` generates 120x160 grayscale examples of clear routes, uncertain routes, and junctions with simulated camera noise.

### CNN

```text
120x160x1
  -> Conv2D(16) -> MaxPool
  -> Conv2D(32) -> MaxPool
  -> Conv2D(64) -> MaxPool
  -> Flatten -> Dense(64) -> Dropout(0.3)
  -> Dense(3, softmax)
```

The script trains for five epochs, runs inference on a test frame, and visualizes nine predictions.

**Important limitations:** the current `__main__` block contains a hard-coded Windows dataset path. Change it. Also, a sequential 80/20 frame split can leak near-duplicate temporal information; serious evaluation should split by recording/session.

Run after fixing the dataset path:

```bash
python cnn.py
```

## `diff_drive.py` — vehicle simulation

Simulates the robot as a differential-drive/unicycle model with:

- 180 mm wheel spacing.
- `(x, y, theta)` pose.
- 50 Hz simulation loop.
- Proportional angular control.
- Route waypoints and synthetic mission markers.
- Slower travel for large heading error.
- Creep-speed docking.

Pose update:

```text
x     += v*cos(theta)*dt
y     += v*sin(theta)*dt
theta += omega*dt
```

The simulation creates curved route sections with a quadratic Bezier curve and switches route segments when mock markers are encountered.

Run:

```bash
python diff_drive.py
```

It prints mission/marker events and plots the simulated trajectory.

## `mission_fsm.py` — mission finite-state machine

Represents the delivery operation explicitly as:

```text
IDLE_BASE
LOAD_AND_LOCK
FOLLOW_ROUTE
JUNCTION
DOCK
UNLOCK_WAIT
CLOSE_AND_RETURN
COMPLETE
OBSTACLE_WAIT
FAULT
```

It consumes a `Telemetry` object containing destination, cargo, lid, route, dock, obstacle and fault conditions.

Global conditions are checked first:

- `fault_flag` -> `FAULT`.
- Obstacle while moving -> `OBSTACLE_WAIT`.

The release stage is interlocked. The mock requires expected marker, valid dock pose, fresh route, stopped wheels, matching mission, cargo present, healthy load sensing and no fault before simulating latch release.

Run:

```bash
python mission_fsm.py
```

The built-in runner simulates a complete mission using mocked telemetry.

## `protocol.py` — binary wire protocol

Defines the host/controller packet format.

### Motion command contains

- magic bytes
- protocol version
- message type
- session ID
- sequence number
- source frame ID
- host timestamp
- linear velocity
- angular velocity
- flags
- CRC16

### Telemetry contains

- session ID
- acknowledged sequence
- controller timestamp
- left/right wheel speeds
- battery voltage
- ultrasonic range
- payload weight
- lid/latch state
- fault flags
- CRC16

The parser rejects bad size, CRC, magic/version, NaN/Infinity and out-of-range motion commands. Current limits are ±0.6 m/s linear and ±2.5 rad/s angular.

**Security note:** CRC detects corruption; it does **not** authenticate the sender. If the physical system uses Wi-Fi for safety-critical release commands, use authentication/MAC and session/replay protection as specified by the engineering design.

## `host_link_manager.py` — host communications manager

Host-side command and telemetry handling.

`send_motion_command()` increments the sequence, timestamps the command, serializes it, and places it in the outbound queue.

`process_incoming_stream()` buffers bytes, validates complete telemetry frames, and stores the latest valid telemetry snapshot behind a thread lock.

The queues are transport-neutral and can later be replaced by USB serial or another local transport.

## `esp32_mock_controller.py` — embedded-controller simulation

This is **not ESP32 firmware**. It models the behaviour expected from the physical controller.

It:

- validates incoming packets;
- checks session and sequence order;
- rejects stale commands;
- handles the emergency-stop flag;
- renews a 200 ms command lease;
- runs a simulated 50 Hz controller tick;
- stops for an obstacle below 30 cm;
- converts `(v, omega)` to left/right wheel targets;
- emits telemetry and fault bits.

The key design principle is: **the host requests motion; the local controller can veto it.**

## `test_link_bench.py` — communications/safety test harness

Runs four tests:

1. Nominal command/telemetry exchange.
2. Host freeze for about 240 ms to exercise watchdog expiry.
3. Corrupt bytes and NaN angular velocity injection.
4. Obstacle override with range forced to 22 cm.

Run:

```bash
python test_link_bench.py
```

This is the best first script to run when reviewing the repository because it exercises failure behaviour rather than only the happy path.

---

# Local setup

## Requirements

Python 3.10+ recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install simulation/perception dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy matplotlib scikit-learn opencv-contrib-python
```

For `cnn.py`:

```bash
python -m pip install tensorflow
```

Verify OpenCV and ArUco:

```bash
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"
```

Use one OpenCV wheel family in the environment. The project guide specifically recommends the contrib build for ArUco support.

---

# Quick start

No hardware required:

```bash
python mission_fsm.py
python test_link_bench.py
python diff_drive.py
```

Then, with a webcam:

```bash
python route_vision2.py
```

Finally, after configuring a real dataset:

```bash
python cnn.py
```

---

# Recorded-video workflow

Use recorded video before moving hardware:

```python
from route_vision2 import VisionPipeline

pipeline = VisionPipeline(source="test_run.mp4")
pipeline.run()
```

Recommended workflow:

```text
record track run
      -> replay video
      -> tune perception
      -> inspect failure cases
      -> test on stationary hardware
      -> test moving hardware
```

---

# How to present this in an autonomous-delivery assessment

I would describe the repository like this:

> This repository is my software architecture and rapid-prototyping layer for an autonomous delivery vehicle. I split the system into perception, mission logic, motion control and an independently enforcing embedded safety layer.
>
> The host interprets camera observations and mission context and sends short-lived velocity commands. The controller validates those commands, checks sequence and freshness, enforces the motion envelope, and can independently stop the vehicle for an obstacle, stale command or emergency-stop condition.
>
> I start perception with classical OpenCV and ArUco because the intermediate signals are inspectable and easy to debug. The ML components are experiments for route confidence/state classification rather than the only safety-critical decision path.
>
> The delivery mission is an explicit finite-state machine. That makes loading, junctions, docking, release, return and faults explicit and testable instead of scattering them through navigation code.
>
> Finally, I wrote a binary protocol and failure-injection harness because an autonomous system has to be tested against bad inputs and missing inputs, not just shown working when everything is perfect.

That is the credible framing. Do **not** call this a finished autonomous robot. It is a software prototype/pre-hardware integration stack.

---

# Current capabilities vs. claims

## Demonstrated in software

- Differential-drive model.
- Closed-loop route simulation.
- ArUco detection.
- Classical route extraction.
- Lightweight route-state classifier architecture.
- CNN route-state experiment.
- Mission FSM.
- Binary command/telemetry protocol.
- CRC, sequence, session, freshness and range validation.
- Mock watchdog and obstacle override.
- Failure-injection tests.

## Not demonstrated by this repository

- Real motor control.
- Real encoder PID.
- Real camera calibration.
- Physical docking accuracy.
- Physical emergency braking distance.
- Real load-cell calibration.
- Real battery/current behaviour.
- Measured end-to-end latency.
- Repeated physical mission acceptance trials.

The companion guide explicitly distinguishes estimates/design pseudocode from measured hardware evidence.

---

# Recommended next engineering steps

1. Add a single application entry point.
2. Move wheel geometry, thresholds, camera settings, serial parameters and limits into configuration.
3. Replace hard-coded ML training points with real labeled data.
4. Replace inferred DonkeyCar labels with independently verified annotations.
5. Split datasets by session/route/run.
6. Replace placeholder camera intrinsics with real calibration.
7. Add `pytest` tests for protocol parsing, watchdog expiry, obstacle stops and FSM transitions.
8. Implement the real ESP32 firmware against the same protocol.
9. Add real encoder speed PI/PID control.
10. Measure end-to-end latency, braking distance, docking error, junction success and payload behaviour on the actual track.

The engineering guide recommends the same staged approach: recorded-video testing, synthetic sensor/FSM tests, ESP32 loopback with wheels raised, low-speed straight driving, junction/docking tests without unlock, empty-bay missions, payload missions, then disturbed/fault testing.

---

# Project status

**Software proof-of-concept / pre-hardware integration.**

The repository's strongest point is not that it already drives a finished robot. It is that the important interfaces and failure modes are being made explicit before hardware integration.
