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
    true_position = calculate_riemann_sum(true_velocity, dt)
    estimated_position = calculate_riemann_sum(measured_velocity, dt)

    # Final Analysis Output
    error = estimated_position[-1] - true_position[-1]
    print(f"True Distance:     {true_position[-1]:.2f} m")
    print(f"Calculated Dist:   {estimated_position[-1]:.2f} m")
    print(f"Final Error:       {error:.2f} m")

    # 5. Visualization

    # Increase base font size for everything (ticks, labels, legends)
    plt.rcParams.update({"font.size": 20})
    # specific settings for titles if you want them even bigger
    plt.rcParams.update({"axes.labelsize": 25})

    # Create figure 1 with 16:9 aspect ratio
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

    ax1.plot(t, true_velocity, "g--", linewidth=8, label="Velocidade real")
    ax1.plot(
        t,
        measured_velocity,
        "r-",
        linewidth=6,
        alpha=0.6,
        label="Leituras do sensor\nde velocidade",
    )
    ax1.set_ylabel("Velocidade (m/s)")
    ax1.set_xlabel("Tempo (s)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    file1 = "results/velocity_plot.png"
    fig1.savefig(file1, dpi=300)

    # Create figure 2 with 16:9 aspect ratio
    fig2, ax2 = plt.subplots(figsize=(10, 5.625), dpi=100)

    ax2.plot(t, true_position, "g--", linewidth=6, label="Posição real")
    ax2.plot(t, estimated_position, "b-", linewidth=6, label="Posição calculada")
    ax2.fill_between(
        t,
        estimated_position,
        true_position,
        color="red",
        alpha=0.8,
        label="Erro de integração",
    )
    ax2.set_ylabel("Posição (m)")
    ax2.set_xlabel("Tempo (s)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()

    # Save Figure 2
    file2 = "results/position_plot.png"
    fig2.savefig(file2, dpi=300)
    print(f"Saved {file2}")

    # This will now pop up two separate windows
    plt.show()


if __name__ == "__main__":
    main()
