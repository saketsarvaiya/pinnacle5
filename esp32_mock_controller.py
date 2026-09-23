import collections
import time
from protocol import (
    CMD_SIZE,
    MotionCommand,
    TelemetryFrame,
    parse_and_validate_command,
)

COMMAND_LEASE_MS = 200  # Hard timeout: motor drive zeroed if lease expires
WHEEL_BASE_M = 0.18  # 180 mm track width


class ESP32ControllerMock:

    def __init__(self, session_id: int):
        self.session_id = session_id
        self.last_accepted_seq = 0
        self.active_lease_expiry = 0
        self.target_v = 0.0
        self.target_omega = 0.0
        self.v_left_measured = 0.0
        self.v_right_measured = 0.0

        # Sensor Mock States
        self.lid_closed = True
        self.latch_locked = True
        self.battery_voltage = 12.4
        self.ultrasonic_distance_cm = 50.0
        self.load_cell_g = 502.0

        # Internal Safety Registers
        self.armed = True
        self.estop = False
        self.lease_expired = False
        self.obstacle_detected = False

        self._inbound_buffer = bytearray()

    def receive_bytes(self, chunk: bytes) -> list[str]:
        logs = []
        self._inbound_buffer.extend(chunk)

        while len(self._inbound_buffer) >= CMD_SIZE:
            candidate = bytes(self._inbound_buffer[:CMD_SIZE])
            valid, reason, cmd = parse_and_validate_command(candidate)

            if not valid:
                # Slide by one byte to recover frame sync on corrupt stream
                del self._inbound_buffer[0]
                logs.append(f"[ESP32 REJECT] Sync dropped: {reason}")
                continue

            del self._inbound_buffer[:CMD_SIZE]
            accepted, ack_reason = self._process_command(cmd)
            logs.append(
                f"[ESP32 INBOUND] Seq={cmd.seq} | Accepted={accepted} ({ack_reason})"
            )

        return logs

    def _process_command(self, cmd: MotionCommand) -> tuple[bool, str]:
        now_ms = int(time.monotonic() * 1000)

        if cmd.session_id != self.session_id:
            return False, "Unknown Session ID"

        if cmd.seq <= self.last_accepted_seq:
            return (
                False,
                f"Out of order sequence: {cmd.seq} <= {self.last_accepted_seq}",
            )

        # Check transit latency (freshness guard)
        transit_delay = now_ms - cmd.host_timestamp_ms
        if transit_delay > COMMAND_LEASE_MS:
            return False, f"Packet arrived stale ({transit_delay} ms old)"

        if cmd.flags & 0x02:  # Emergency Stop requested
            self.estop = True
            self.target_v = 0.0
            self.target_omega = 0.0
            return True, "E-Stop Latched"

        # Valid setpoint: renew command lease
        self.last_accepted_seq = cmd.seq
        self.target_v = cmd.v
        self.target_omega = cmd.omega
        self.active_lease_expiry = now_ms + COMMAND_LEASE_MS
        self.lease_expired = False
        return True, "Lease Renewed"

    def tick_50hz(self) -> bytes:
        """Runs every 20ms: updates physics mock, checks safety, outputs telemetry."""
        now_ms = int(time.monotonic() * 1000)

        # Safety Audit 1: Watchdog lease expiry
        if now_ms > self.active_lease_expiry:
            self.lease_expired = True
            self.target_v = 0.0
            self.target_omega = 0.0

        # Safety Audit 2: Obstacle thresholding (< 30 cm)
        if self.ultrasonic_distance_cm < 30.0:
            self.obstacle_detected = True
            self.target_v = 0.0
            self.target_omega = 0.0
        else:
            self.obstacle_detected = False

        # Physics Emulation: Differential drive wheel speed generation
        # v_left = v - (omega * b / 2), v_right = v + (omega * b / 2)
        target_vl = self.target_v - (self.target_omega * WHEEL_BASE_M / 2.0)
        target_vr = self.target_v + (self.target_omega * WHEEL_BASE_M / 2.0)

        # Low-pass filter to simulate motor inertial lag
        self.v_left_measured += 0.25 * (target_vl - self.v_left_measured)
        self.v_right_measured += 0.25 * (target_vr - self.v_right_measured)

        telem = TelemetryFrame(
            session_id=self.session_id,
            ack_seq=self.last_accepted_seq,
            controller_timestamp_ms=now_ms,
            v_left=self.v_left_measured,
            v_right=self.v_right_measured,
            battery_v=self.battery_voltage,
            range_cm=self.ultrasonic_distance_cm,
            weight_g=self.load_cell_g,
            lid_closed=self.lid_closed,
            latch_locked=self.latch_locked,
            fault_lease_expired=self.lease_expired,
            fault_obstacle=self.obstacle_detected,
            fault_low_battery=self.battery_voltage < 10.8,
            fault_estop=self.estop,
        )
        return telem.serialize()