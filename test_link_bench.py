import math
import struct
import time
from esp32_mock_controller import ESP32ControllerMock
from host_link_manager import HostLinkManager


def run_benchmark():
    SESSION_ID = 42
    host = HostLinkManager(session_id=SESSION_ID)
    controller = ESP32ControllerMock(session_id=SESSION_ID)

    print("=" * 65)
    print("STAGE 1: Nominal Closed-Loop Control (v=0.35 m/s, w=0.10 rad/s)")
    print("=" * 65)

    for i in range(1, 6):
        # Host sends motion packet bound to camera frame ID
        packet = host.send_motion_command(v=0.35, omega=0.10, frame_id=100 + i)
        # Transmit to ESP32 over serial channel
        controller_logs = controller.receive_bytes(packet)
        for log in controller_logs:
            print(log)

        # Step ESP32 50 Hz Loop (20 ms interval)
        telem_bytes = controller.tick_50hz()
        host.process_incoming_stream(telem_bytes)

        t = host.get_telemetry_snapshot()
        print(
            f"Host Rx <- AckSeq={t.ack_seq} | vL={t.v_left:.3f} m/s, vR={t.v_right:.3f} m/s | Safe={not t.fault_lease_expired}"
        )
        time.sleep(0.02)

    print("\n" + "=" * 65)
    print("STAGE 2: Watchdog Command Expiry Test (Simulating 250 ms Host Freeze)")
    print("=" * 65)

    print("Host stalls... ESP32 loops without fresh commands")
    for step in range(12):  # 12 * 20ms = 240ms of stall time
        telem_bytes = controller.tick_50hz()
        host.process_incoming_stream(telem_bytes)
        t = host.get_telemetry_snapshot()
        if step % 3 == 0 or t.fault_lease_expired:
            print(
                f"[T+{step*20:03d}ms] LeaseExpired={t.fault_lease_expired} | Target Speed Zeroed | vL={t.v_left:.3f} m/s"
            )
        time.sleep(0.02)

    print("\n" + "=" * 65)
    print("STAGE 3: Malformed Packet / NaN Setpoint Injection Test")
    print("=" * 65)

    # 1. Corrupt byte stream
    corrupt_bytes = b"\xAA\x55\x01\x01\x00\x2A\xFF\xFF\xFF"
    print("Injecting arbitrary byte sequence...")
    logs = controller.receive_bytes(corrupt_bytes)
    for log in logs:
        print(log)

    # 2. NaN float setpoint injection
    print("Injecting NaN angular velocity setpoint...")
    host.seq += 1
    bad_header = struct.pack(
        ">2sBBHIIIffB",
        b"\xAA\x55",
        1,
        1,
        SESSION_ID,
        host.seq,
        999,
        int(time.monotonic() * 1000),
        0.3,
        float("nan"),
        0,
    )
    from protocol import compute_crc16

    bad_packet = bad_header + struct.pack(">H", compute_crc16(bad_header))
    logs = controller.receive_bytes(bad_packet)
    for log in logs:
        print(log)

    print("\n" + "=" * 65)
    print("STAGE 4: Hardware Obstacle Stop Override (< 30 cm)")
    print("=" * 65)

    controller.ultrasonic_distance_cm = 22.0  # Obstacle placed at 22 cm
    packet = host.send_motion_command(v=0.40, omega=0.0, frame_id=201)
    controller.receive_bytes(packet)
    telem_bytes = controller.tick_50hz()
    host.process_incoming_stream(telem_bytes)

    t = host.get_telemetry_snapshot()
    print(
        f"Obstacle Range: {t.range_cm} cm | ObstacleFault: {t.fault_obstacle} | Motor Drive Clamped to Zero: {t.v_left == 0.0 and t.v_right < 0.1}"
    )


if __name__ == "__main__":
    run_benchmark()