import numpy as np
import matplotlib.pyplot as plt


def main():
    print("Hello from demo!")

    # 1. Setup Simulation Parameters
    dt = 0.1  # Sampling time (0.1s)
    total_time = 10  # Seconds
    t = np.arange(0, total_time, dt)

    # 2. Generate "True" World Data (The Perfect Scenario)
    true_velocity = np.zeros_like(t)
    true_velocity[t < 2] = t[t < 2] * 1.0  # Ramp up
    true_velocity[(t >= 2) & (t < 8)] = 2.0  # Cruise
    true_velocity[t >= 8] = 2.0 - (t[t >= 8] - 8)  # Ramp down

    # 3. Add Sensor Noise (The "Real" Scenario)
    # We add random Gaussian noise to simulate a real sensor
    noise_level = 0.2  # Standard deviation of the noise
    sensor_noise = np.random.normal(0, noise_level, len(t))
    measured_velocity = true_velocity + sensor_noise

    # 4. Numerical Integration (Riemann Sum)
    # We integrate the NOISY data, because that's all the robot sees!
    estimated_position = np.zeros_like(t)
    true_position = np.zeros_like(t)

    current_est_pos = 0
    current_true_pos = 0

    for i in range(1, len(t)):
        # Calculate displacement for this step (dx = v * dt)
        dx_est = measured_velocity[i] * dt
        dx_true = true_velocity[i] * dt

        # Accumulate (The Integral)
        current_est_pos += dx_est
        current_true_pos += dx_true

        # Store history
        estimated_position[i] = current_est_pos
        true_position[i] = current_true_pos

    # Final Analysis Output
    error = estimated_position[-1] - true_position[-1]
    print(f"True Distance:     {true_position[-1]:.2f} m")
    print(f"Calculated Dist:   {estimated_position[-1]:.2f} m")
    print(f"Final Error:       {error:.2f} m")

    # 5. Visualization
    plt.figure(figsize=(10, 8))

    # Plot 1: Velocity (Derivative layer)
    plt.subplot(2, 1, 1)
    plt.plot(t, true_velocity, "g--", linewidth=2, label="True Velocity (Ideal)")
    plt.plot(t, measured_velocity, "r-", alpha=0.6, label="Sensor Reading (Noisy)")
    plt.title("Input: Velocity Data (Derivative)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Position (Integral layer)
    plt.subplot(2, 1, 2)
    plt.plot(t, true_position, "g--", linewidth=2, label="True Position")
    plt.plot(
        t,
        estimated_position,
        "b-",
        linewidth=2,
        label="Computed Position (Integration)",
    )
    plt.fill_between(
        t,
        estimated_position,
        true_position,
        color="red",
        alpha=0.8,
        label="Integration Error",
    )
    plt.title("Output: Position (Integral)")
    plt.ylabel("Position (m)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
