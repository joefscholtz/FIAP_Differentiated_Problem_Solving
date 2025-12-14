import numpy as np
import matplotlib.pyplot as plt
from math_utils import calculate_riemann_sum  # Import your new function


def main():
    print("Starting simulation...")

    # 1. Setup Simulation Parameters
    dt = 0.1
    total_time = 10
    t = np.arange(0, total_time, dt)

    # 2. Generate "True" World Data
    true_velocity = np.zeros_like(t)
    true_velocity[t < 2] = t[t < 2] * 1.0
    true_velocity[(t >= 2) & (t < 8)] = 2.0
    true_velocity[t >= 8] = 2.0 - (t[t >= 8] - 8)

    # 3. Add Sensor Noise & Quantization
    noise_level = 0.5
    sensor_noise = np.random.normal(0, noise_level, len(t))
    noisy_signal = true_velocity + sensor_noise

    precision_step = 0.2
    measured_velocity = np.round(noisy_signal / precision_step) * precision_step

    # 4. Numerical Integration (Using the imported function)
    # We call the function twice: once for the perfect data, once for the noisy data
    true_position = calculate_riemann_sum(true_velocity, dt)
    estimated_position = calculate_riemann_sum(measured_velocity, dt)

    # Final Analysis Output
    error = estimated_position[-1] - true_position[-1]
    print(f"True Distance:     {true_position[-1]:.2f} m")
    print(f"Calculated Dist:   {estimated_position[-1]:.2f} m")
    print(f"Final Error:       {error:.2f} m")

    # 5. Visualization
    # Create figure 1 with 16:9 aspect ratio
    fig1, ax1 = plt.subplots(figsize=(10, 5.625), dpi=100)

    ax1.plot(t, true_velocity, "g--", linewidth=2, label="True Velocity (Ideal)")
    ax1.plot(t, measured_velocity, "r-", alpha=0.6, label="Sensor Reading (Noisy)")
    ax1.set_title("Input: Velocity Data (Derivative)")
    ax1.set_ylabel("Velocity (m/s)")
    # Added X-label here since it's now a standalone plot
    ax1.set_xlabel("Time (s)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    file1 = "velocity_plot.png"
    fig1.savefig(file1, dpi=300)

    # Create figure 2 with 16:9 aspect ratio
    fig2, ax2 = plt.subplots(figsize=(10, 5.625), dpi=100)

    ax2.plot(t, true_position, "g--", linewidth=2, label="True Position")
    ax2.plot(t, estimated_position, "b-", linewidth=2, label="Computed Position")
    ax2.fill_between(
        t,
        estimated_position,
        true_position,
        color="red",
        alpha=0.8,
        label="Integration Error",
    )
    ax2.set_title("Output: Position (Integral)")
    ax2.set_ylabel("Position (m)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()

    # Save Figure 2
    file2 = "position_plot.png"
    fig2.savefig(file2, dpi=300)
    print(f"Saved {file2}")

    # This will now pop up two separate windows
    plt.show()


if __name__ == "__main__":
    main()
