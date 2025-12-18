import numpy as np


def calculate_riemann_sum(data_list, dt):
    integral_result = []
    current_area = 0.0
    for value in data_list:
        rectangle_area = value * dt
        current_area += rectangle_area
        integral_result.append(current_area)

    return integral_result


def calculate_riemann_sum2(data_list, dt):
    """
    Computes the integral using a standard Python loop.
    This demonstrates the concept of 'Accumulation'.

    Args:
        data_list (list or array): The input velocity values
        dt (float): The time step

    Returns:
        list: The position values (Integration result)
    """
    # 1. Initialize an empty list to hold the result
    integral_result = []

    # 2. Initialize the "Accumulator" variable
    # This represents the area collected so far
    current_area = 0.0

    # 3. Iterate through every single data point
    for velocity_value in data_list:

        # Step A: Calculate area of the tiny rectangle (base * height)
        # Mathematical equivalent: dx = v * dt
        rectangle_area = velocity_value * dt

        # Step B: Add it to the total (Integration)
        # Mathematical equivalent: S = S + dx
        current_area += rectangle_area

        # Step C: Save the current state to our list
        integral_result.append(current_area)

    return integral_result


def calculate_riemann_sum_np(data_array, dt):
    """
    Computes the indefinite integral of an array using the Riemann Sum method using numpy .

    Formula: Position[i] = Sum(Velocity[0...i] * dt)

    Args:
        data_array (np.array): The input signal (e.g., velocity)
        dt (float): The sampling time interval

    Returns:
        np.array: The integrated signal (e.g., position)
    """
    # np.cumsum calculates the cumulative sum of elements.
    # [v0, v1, v2] -> [v0, v0+v1, v0+v1+v2]
    # We multiply by dt because Area = height * width
    integrated_array = np.cumsum(data_array) * dt

    return integrated_array
