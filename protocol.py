import math
import struct
import time
from dataclasses import dataclass

CMD_MAGIC = b"\xAA\x55"
TELEM_MAGIC = b"\x55\xAA"
PROTOCOL_VERSION = 1

# Command Payload Layout (Big-Endian):
# Magic: 2B (0xAA55)
# Version: 1B (uint8)
# Msg Type: 1B (uint8) -> 1: Motion, 2: Unlock, 3: Arm/Disarm
# Session ID: 2B (uint16)
# Sequence: 4B (uint32)
# Frame ID: 4B (uint32)
# Host Monotonic Time (ms): 4B (uint32)
# Velocity v (m/s): 4B (float32)
# Angular velocity omega (rad/s): 4B (float32)
# Flags: 1B (uint8) -> Bit 0: Arm, Bit 1: Emergency Stop, Bit 2: Force Coast
# CRC16: 2B (uint16)
CMD_FORMAT = ">2sBBHIIIffBH"
CMD_SIZE = struct.calcsize(CMD_FORMAT)

# Telemetry Payload Layout (Big-Endian):
# Magic: 2B (0x55AA)
# Version: 1B (uint8)
# Session ID: 2B (uint16)
# Ack Sequence: 4B (uint32)
# Controller Monotonic Time (ms): 4B (uint32)
# Measured v_left (m/s): 4B (float32)
# Measured v_right (m/s): 4B (float32)
# Battery Voltage (V): 4B (float32)
# Ultrasonic Range (cm): 4B (float32)
# Payload Weight (g): 4B (float32)
# Hardware State: 1B (uint8) -> Bit 0: Lid Closed, Bit 1: Latch Locked
# Fault State: 1B (uint8) -> Bit 0: Stale Lease, Bit 1: Obstacle, Bit 2: Low Bat, Bit 3: Estop
# CRC16: 2B (uint16)
TELEM_FORMAT = ">2sBHIIfffffBBH"
TELEM_SIZE = struct.calcsize(TELEM_FORMAT)

MAX_LINEAR_VELOCITY = 0.6  # m/s
MAX_ANGULAR_VELOCITY = 2.5  # rad/s


def compute_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


@dataclass
class MotionCommand:
    msg_type: int
    session_id: int
    seq: int
    frame_id: int
    host_timestamp_ms: int
    v: float
    omega: float
    flags: int = 0

    def serialize(self) -> bytes:
        header_and_body = struct.pack(
            ">2sBBHIIIffB",
            CMD_MAGIC,
            PROTOCOL_VERSION,
            self.msg_type,
            self.session_id,
            self.seq,
            self.frame_id,
            self.host_timestamp_ms,
            float(self.v),
            float(self.omega),
            self.flags,
        )
        crc = compute_crc16(header_and_body)
        return header_and_body + struct.pack(">H", crc)


@dataclass
class TelemetryFrame:
    session_id: int
    ack_seq: int
    controller_timestamp_ms: int
    v_left: float
    v_right: float
    battery_v: float
    range_cm: float
    weight_g: float
    lid_closed: bool
    latch_locked: bool
    fault_lease_expired: bool
    fault_obstacle: bool
    fault_low_battery: bool
    fault_estop: bool

    def serialize(self) -> bytes:
        hw_byte = (1 if self.lid_closed else 0) | (
            (1 if self.latch_locked else 0) << 1
        )
        fault_byte = (
            (1 if self.fault_lease_expired else 0)
            | ((1 if self.fault_obstacle else 0) << 1)
            | ((1 if self.fault_low_battery else 0) << 2)
            | ((1 if self.fault_estop else 0) << 3)
        )
        body = struct.pack(
            ">2sBHIIfffffBB",
            TELEM_MAGIC,
            PROTOCOL_VERSION,
            self.session_id,
            self.ack_seq,
            self.controller_timestamp_ms,
            float(self.v_left),
            float(self.v_right),
            float(self.battery_v),
            float(self.range_cm),
            float(self.weight_g),
            hw_byte,
            fault_byte,
        )
        crc = compute_crc16(body)
        return body + struct.pack(">H", crc)


def parse_and_validate_command(
    raw_bytes: bytes,
) -> tuple[bool, str, MotionCommand | None]:
    if len(raw_bytes) != CMD_SIZE:
        return (
            False,
            f"Invalid size: expected {CMD_SIZE}, received {len(raw_bytes)}",
            None,
        )

    expected_crc = struct.unpack(">H", raw_bytes[-2:])[0]
    calculated_crc = compute_crc16(raw_bytes[:-2])
    if expected_crc != calculated_crc:
        return (
            False,
            f"CRC mismatch: expected {expected_crc:04X}, calculated {calculated_crc:04X}",
            None,
        )

    magic, version, msg_type, session, seq, frame_id, t_ms, v, omega, flags = (
        struct.unpack(">2sBBHIIIffB", raw_bytes[:-2])
    )

    if magic != CMD_MAGIC:
        return False, "Invalid command magic header", None
    if version != PROTOCOL_VERSION:
        return (
            False,
            f"Protocol version mismatch: {version} != {PROTOCOL_VERSION}",
            None,
        )

    # Reject non-finite values (NaN / Inf) per specification
    if math.isnan(v) or math.isinf(v) or math.isnan(omega) or math.isinf(omega):
        return False, "NaN or Infinity detected in motion vector", None

    # Kinematic envelope clamp
    if abs(v) > MAX_LINEAR_VELOCITY or abs(omega) > MAX_ANGULAR_VELOCITY:
        return (
            False,
            f"Setpoints out of bounds: v={v:.2f} m/s, w={omega:.2f} rad/s",
            None,
        )

    cmd = MotionCommand(
        msg_type=msg_type,
        session_id=session,
        seq=seq,
        frame_id=frame_id,
        host_timestamp_ms=t_ms,
        v=v,
        omega=omega,
        flags=flags,
    )
    return True, "OK", cmd


def parse_telemetry(
    raw_bytes: bytes,
) -> tuple[bool, str, TelemetryFrame | None]:
    if len(raw_bytes) != TELEM_SIZE:
        return False, f"Size error: got {len(raw_bytes)}", None

    expected_crc = struct.unpack(">H", raw_bytes[-2:])[0]
    calculated_crc = compute_crc16(raw_bytes[:-2])
    if expected_crc != calculated_crc:
        return False, "Telemetry CRC mismatch", None

    (
        magic,
        version,
        session,
        ack_seq,
        t_ms,
        vl,
        vr,
        bat,
        rng,
        wt,
        hw_byte,
        fault_byte,
    ) = struct.unpack(">2sBHIIfffffBB", raw_bytes[:-2])

    if magic != TELEM_MAGIC:
        return False, "Invalid telemetry magic", None

    telem = TelemetryFrame(
        session_id=session,
        ack_seq=ack_seq,
        controller_timestamp_ms=t_ms,
        v_left=vl,
        v_right=vr,
        battery_v=bat,
        range_cm=rng,
        weight_g=wt,
        lid_closed=bool(hw_byte & 0x01),
        latch_locked=bool(hw_byte & 0x02),
        fault_lease_expired=bool(fault_byte & 0x01),
        fault_obstacle=bool(fault_byte & 0x02),
        fault_low_battery=bool(fault_byte & 0x04),
        fault_estop=bool(fault_byte & 0x08),
    )
    return True, "OK", telem