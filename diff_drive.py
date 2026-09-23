import math
import numpy as np
import matplotlib.pyplot as plt

class RealisticRoverSimulator:
    def __init__(self, wheel_base=0.18):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.b = wheel_base
        
        self.history_x = [self.x]
        self.history_y = [self.y]
        
        # Proportional gain for the visual line-following controller
        self.Kp = 2.5 

    def kinematics_update(self, v, omega, dt):
        """Updates global pose using differential drive equations[cite: 4]."""
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt
        
        # Normalize theta to [-pi, pi]
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        
        self.history_x.append(self.x)
        self.history_y.append(self.y)
        
    def mock_cv_pipeline(self, active_route):
        """
        Simulates the OpenCV line centroid tracking.
        Finds a lookahead point on the track and calculates the heading error.
        """
        if len(active_route) == 0:
            return 0.0, 0.0 # No route visible

        # Target the next immediate waypoint on the active route
        target_x, target_y = active_route[0]
        dx = target_x - self.x
        dy = target_y - self.y
        distance_to_target = math.hypot(dx, dy)
        
        # Calculate heading error (simulating lateral centroid offset)
        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - self.theta
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        
        # If close to the waypoint, pop it so the CV "looks" at the next segment
        if distance_to_target < 0.15 and len(active_route) > 1:
            active_route.pop(0)
            
        return heading_error, distance_to_target

def generate_smooth_curve(start, end, control, num_points=10):
    """Generates a quadratic Bezier curve to simulate smooth tape corners."""
    t = np.linspace(0, 1, num_points)
    x = (1-t)**2 * start[0] + 2*(1-t)*t * control[0] + t**2 * end[0]
    y = (1-t)**2 * start[1] + 2*(1-t)*t * control[1] + t**2 * end[1]
    return list(zip(x, y))

def run_realistic_simulation():
    sim = RealisticRoverSimulator()
    dt = 0.05 # 50Hz control loop to match ESP32 scheduling
    state = "IDLE_BASE"
    
    # 1. Define the physical track (Tape on the floor)
    # Segment 1: Base to Junction 1
    route_seg1 = [(0.5, 0.0), (1.0, 0.0), (1.5, 0.0)]
    # Segment 2: Junction 1 curve and straight to Intermediate Stop
    route_seg2 = generate_smooth_curve((1.5, 0.0), (2.0, 0.5), (2.0, 0.0)) + [(2.0, 1.0), (2.0, 1.5), (2.0, 2.2)]
    # Segment 3: Intermediate Stop to Junction 2 curve and approach to Dock
    route_seg3 = generate_smooth_curve((2.0, 1.9), (1.5, 2.0), (2.0, 2.0)) + [(1.0, 2.0), (0.4, 2.0), (0.0, 2.0)]
    
    active_route = route_seg1.copy()
    
    # Define physical ArUco marker locations (x, y)
    markers = {
        "J1": (1.4, 0.0),      # Triggers turn onto Seg 2
        "STOP1": (2.0, 1.4),   # Triggers Intermediate Stop
        "J2": (1.9, 1.9),      # Triggers turn onto Seg 3
        "DOCK": (0.45, 2.0)    # Triggers Creep and final Dock[cite: 1]
    }
    
    wait_timer = 0.0
    
    print("Starting Closed-Loop Vision Simulator...\n")
    
    for step in range(600): # Max 30 seconds
        current_time = step * dt
        v, omega = 0.0, 0.0
        
        # --- SENSE: Mock CNN Marker Detection (Proximity check) ---
        detected_marker = None
        for name, (mx, my) in list(markers.items()):
            if math.hypot(mx - sim.x, my - sim.y) < 0.15:
                detected_marker = name
                del markers[name] # "Debounce" marker so it isn't triggered twice[cite: 1]
                break

        # --- THINK: Mission FSM ---
        if step == 0:
            state = "FOLLOW_ROUTE"
            print(f"[{current_time:.2f}s] FSM: -> FOLLOW_ROUTE (Departing Base)")
            
        elif detected_marker == "J1":
            print(f"[{current_time:.2f}s] CNN: Marker 'J1' Detected.")
            print(f"[{current_time:.2f}s] FSM: -> JUNCTION (Switching to Route Segment 2)")
            active_route = route_seg2.copy()
            
        elif detected_marker == "STOP1":
            print(f"\n[{current_time:.2f}s] CNN: Marker 'STOP1' Detected.")
            print(f"[{current_time:.2f}s] FSM: -> INTERMEDIATE_WAIT (Pausing for cargo check)")
            state = "INTERMEDIATE_WAIT"
            wait_timer = 2.0 # Wait for 2 seconds
            
        elif detected_marker == "J2":
            print(f"\n[{current_time:.2f}s] CNN: Marker 'J2' Detected.")
            print(f"[{current_time:.2f}s] FSM: -> JUNCTION (Switching to Route Segment 3)")
            active_route = route_seg3.copy()
            
        elif detected_marker == "DOCK":
            print(f"\n[{current_time:.2f}s] CNN: Marker 'DOCK' Detected.")
            print(f"[{current_time:.2f}s] FSM: -> DOCK (Creeping to target pose)")
            state = "DOCK"
            
        # Handle state logic
        if state == "INTERMEDIATE_WAIT":
            v, omega = 0.0, 0.0
            wait_timer -= dt
            if wait_timer <= 0:
                print(f"[{current_time:.2f}s] FSM: -> FOLLOW_ROUTE (Resuming mission)")
                state = "FOLLOW_ROUTE"
                
        elif state in ["FOLLOW_ROUTE", "DOCK"]:
            # --- THINK: Visual Controller Loop ---
            heading_error, dist = sim.mock_cv_pipeline(active_route)
            
            # 1. Set Linear Velocity (v)
            if state == "DOCK":
                v = 0.1 # Creep speed for docking[cite: 1]
                if dist < 0.05 and len(active_route) <= 1:
                    print(f"[{current_time:.2f}s] FSM: -> UNLOCK_WAIT (Wheels stopped, stable pose)[cite: 1]")
                    state = "COMPLETE"
            else:
                # Slow down if the heading error is large (e.g., sharp curves)[cite: 1]
                v = 0.4 if abs(heading_error) < 0.3 else 0.2
                
            # 2. Set Angular Velocity (omega) using Proportional Control[cite: 1]
            omega = sim.Kp * heading_error
            
            # Apply mechanical limits to prevent spinning out
            omega = max(-1.5, min(1.5, omega))
            
        elif state == "COMPLETE":
            v, omega = 0.0, 0.0
            break

        # --- ACT: Kinematics Update ---
        sim.kinematics_update(v, omega, dt)

    # Render Plot
    print("\nSimulation complete. Rendering closed-loop trajectory...")
    plt.figure(figsize=(8, 6))
    plt.plot(sim.history_x, sim.history_y, 'b-', linewidth=2, label='Closed-Loop Robot Trajectory')
    
    # Plot markers for visual reference
    plt.plot(0, 0, 'go', markersize=10, label='Base Station')
    plt.plot(1.4, 0.0, 'ms', markersize=8, label='J1 Marker')
    plt.plot(2.0, 1.4, 'cs', markersize=8, label='Intermediate Stop')
    plt.plot(1.9, 1.9, 'ms', markersize=8, label='J2 Marker')
    plt.plot(0.0, 2.0, 'ro', markersize=10, label='Delivery Dock')
    
    plt.title('Realistic Closed-Loop Rover Trajectory')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    run_realistic_simulation()