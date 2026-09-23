import queue
import threading
import time
from protocol import (
    TELEM_SIZE,
    MotionCommand,
    TelemetryFrame,
    parse_telemetry,
)


class HostLinkManager:

    def __init__(self, session_id: int):
        self.session_id = session_id
        self.seq = 0
        self.latest_telemetry: TelemetryFrame | None = None
        self._lock = threading.Lock()

        # Telemetry inbound queue (bounded to prevent frame backlog)
        self._rx_buffer = bytearray()
        self.outbound_wire: queue.Queue[bytes] = queue.Queue()
        self.inbound_wire: queue.Queue[bytes] = queue.Queue()

    def send_motion_command(
        self, v: float, omega: float, frame_id: int, estop: bool = False
    ) -> bytes:
        self.seq += 1
        now_ms = int(time.monotonic() * 1000)
        flags = 0x02 if estop else 0x01

        cmd = MotionCommand(
            msg_type=1,
            session_id=self.session_id,
            seq=self.seq,
            frame_id=frame_id,
            host_timestamp_ms=now_ms,
            v=v,
            omega=omega,
            flags=flags,
        )
        packet = cmd.serialize()
        self.outbound_wire.put(packet)
        return packet

    def process_incoming_stream(self, raw_bytes: bytes):
        self._rx_buffer.extend(raw_bytes)
        while len(self._rx_buffer) >= TELEM_SIZE:
            candidate = bytes(self._rx_buffer[:TELEM_SIZE])
            valid, reason, telem = parse_telemetry(candidate)
            if not valid:
                del self._rx_buffer[0]
                continue

            del self._rx_buffer[:TELEM_SIZE]
            with self._lock:
                self.latest_telemetry = telem

    def get_telemetry_snapshot(self) -> TelemetryFrame | None:
        with self._lock:
            return self.latest_telemetry