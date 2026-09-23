import time
from enum import Enum, auto
from dataclasses import dataclass

class State(Enum):
    IDLE_BASE = auto()
    LOAD_AND_LOCK = auto()
    FOLLOW_ROUTE = auto()
    JUNCTION = auto()
    DOCK = auto()
    UNLOCK_WAIT = auto()
    CLOSE_AND_RETURN = auto()
    COMPLETE = auto()
    OBSTACLE_WAIT = auto()
    FAULT = auto()

@dataclass
class Telemetry:
    """Mock telemetry mapping to the required sensing and logic gates."""
    base_id_known: bool = True
    destination_valid: bool = False
    sensors_healthy: bool = True
    cargo_present: bool = False
    lid_closed: bool = False
    operator_start: bool = False
    route_fresh: bool = False
    expected_marker_seen: bool = False
    dock_pose_valid: bool = False
    wheels_stopped: bool = False
    active_mission_match: bool = False
    load_sensor_healthy: bool = True
    obstacle_detected: bool = False
    fault_flag: bool = False

class MissionFSM:
    def __init__(self):
        self.state = State.IDLE_BASE
        self.mission_id = None
        self.log = []

    def update(self, t: Telemetry):
        # Global fault interrupt
        if t.fault_flag and self.state != State.FAULT:
            self.transition(State.FAULT, "Hardware or software fault detected.")
            return

        # Global obstacle interrupt (if moving)
        if t.obstacle_detected and self.state in [State.FOLLOW_ROUTE, State.JUNCTION, State.DOCK]:
            self.transition(State.OBSTACLE_WAIT, "Obstacle in path.")
            return

        # State transition evaluation
        if self.state == State.IDLE_BASE:
            if t.base_id_known and t.destination_valid and t.sensors_healthy and t.cargo_present:
                self.transition(State.LOAD_AND_LOCK, "Mission parameters valid. Cargo detected.")

        elif self.state == State.LOAD_AND_LOCK:
            if t.cargo_present and t.lid_closed and t.operator_start:
                self.mission_id = int(time.time()) # Generate unique mission ID
                t.active_mission_match = True
                self.transition(State.FOLLOW_ROUTE, "Bay locked. Departing.")

        elif self.state == State.FOLLOW_ROUTE:
            if t.expected_marker_seen:
                self.transition(State.JUNCTION, "Approaching junction marker.")
            elif t.dock_pose_valid: # Simplification for mock
                self.transition(State.DOCK, "Approaching destination dock.")

        elif self.state == State.JUNCTION:
            if t.route_fresh:
                self.transition(State.FOLLOW_ROUTE, "Turn complete, route reacquired.")

        elif self.state == State.DOCK:
            if t.expected_marker_seen and t.dock_pose_valid and t.wheels_stopped:
                self.transition(State.UNLOCK_WAIT, "Dock pose achieved.")

        elif self.state == State.UNLOCK_WAIT:
            # The strict release interlocked transaction requirement
            release_authorized = (
                t.expected_marker_seen and 
                t.dock_pose_valid and 
                t.route_fresh and 
                t.wheels_stopped and 
                t.active_mission_match and 
                t.cargo_present and 
                t.load_sensor_healthy and 
                not t.fault_flag
            )
            if release_authorized:
                print("[HARDWARE ACTION] Latch Servo Released")
                t.lid_closed = False # Operator opens lid
                t.cargo_present = False # Operator removes load
                
            if not t.cargo_present and t.lid_closed:
                self.transition(State.CLOSE_AND_RETURN, "Cargo removed. Relocked.")

        elif self.state == State.CLOSE_AND_RETURN:
            if t.base_id_known and t.dock_pose_valid and t.wheels_stopped:
                self.transition(State.COMPLETE, "Returned to base safely.")

        elif self.state == State.OBSTACLE_WAIT:
            if not t.obstacle_detected and t.sensors_healthy:
                self.transition(State.FOLLOW_ROUTE, "Path cleared. Resuming.")

        elif self.state == State.FAULT:
            if t.operator_start and t.sensors_healthy:
                self.transition(State.IDLE_BASE, "Fault cleared by operator. Resetting.")

    def transition(self, new_state: State, reason: str):
        print(f"[STATE CHANGE] {self.state.name} -> {new_state.name} | {reason}")
        self.state = new_state
        self.log.append((time.time(), new_state.name, reason))

# --- MOCK RUNNER (For Prototyping) ---
if __name__ == "__main__":
    fsm = MissionFSM()
    telemetry = Telemetry()

    print(f"Initial State: {fsm.state.name}")
    print("--- Simulating Mission Start ---")
    
    # Simulate Operator defining a destination and loading cargo
    telemetry.destination_valid = True
    telemetry.cargo_present = True
    fsm.update(telemetry)
    
    # Simulate closing the lid and starting
    telemetry.lid_closed = True
    telemetry.operator_start = True
    fsm.update(telemetry)
    
    # Simulate driving and finding a junction
    telemetry.operator_start = False 
    telemetry.route_fresh = True
    telemetry.expected_marker_seen = True
    fsm.update(telemetry)
    
    # Simulate clearing the junction
    telemetry.expected_marker_seen = False
    fsm.update(telemetry)
    
    # Simulate finding the destination dock
    telemetry.dock_pose_valid = True
    telemetry.expected_marker_seen = True
    telemetry.wheels_stopped = True
    fsm.update(telemetry)
    
    # The FSM will now authorize unlock and simulate cargo removal
    fsm.update(telemetry)
    
    # Simulate operator closing the lid after taking the package
    telemetry.lid_closed = True
    fsm.update(telemetry)
    
    # Simulate returning to base
    telemetry.base_id_known = True
    fsm.update(telemetry)