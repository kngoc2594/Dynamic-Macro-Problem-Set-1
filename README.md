# Dynamic-Macro-Problem-Set-1
# Problem set 3
# c) Set the seed for simulation to 2025. Draw the initial states from the state vector, assuming the initial unconditional distribution is uniform. Simulate and plot the Markov Chain in (b) for 50 periods.
# simulate.py
# ------------
# This code simulates a Markov chain given a grid and transition matrix.

#%% Imports.

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

#%% Simulation function.

def simulate(grid, pmat, T, seed=2025):
    """
    This code simulates a Markov chain given a grid of points from the discretized process and the associated transition matrix.

    Input:
        grid : Grid of discretized points (1D array).
        pmat : Transition probability matrix (2D array).
        T    : Number of periods to simulate.
        seed : Seed for random number generator (default: 2025).

    Output:
        y : Simulated series (1D array).
    """

    # Set seed for reproducibility.
    rng = default_rng(seed)

    #%% Initialize.
    N = len(grid)  # Number of states.
    pi0 = np.ones(N) / N  # Uniform initial distribution.
    state0 = rng.choice(N, p=pi0)  # Draw initial state uniformly.

    #%% Simulate.
    cmat = np.cumsum(pmat, axis=1)  # CDF matrix.
    y = np.zeros(T * 2)  # Container for simulated series.

    for i in range(T * 2):  # Simulation.
        y[i] = grid[state0]  # Current state.
        state1 = cmat[state0, rng.random() <= cmat[state0, :]]  # State next period.
        state0 = np.nonzero(cmat[state0, :] == state1[0])[0][0]  # Update index for next period.

    y = y[T:]  # Burn the first half.

    #%% Output.
    return y

#%% Main script.

if __name__ == "__main__":
    # Given parameters.
    gamma1 = 0.85
    mu = 0.5 / (1 - gamma1)
    sigma_eps = 1
    N = 7
    T = 50

    # State vector.
    z = np.linspace(-3, 3, N)
    grid = mu + z * np.sqrt(3.6036) * np.sqrt(N - 1)

    # Transition matrix.
    p = (1 + gamma1) / 2
    pmat = np.array([
        [0.925, 0.075, 0, 0, 0, 0, 0],
        [0.075, 0.850, 0.075, 0, 0, 0, 0],
        [0, 0.075, 0.850, 0.075, 0, 0, 0],
        [0, 0, 0.075, 0.850, 0.075, 0, 0],
        [0, 0, 0, 0.075, 0.850, 0.075, 0],
        [0, 0, 0, 0, 0.075, 0.850, 0.075],
        [0, 0, 0, 0, 0, 0.075, 0.925]
    ])

    # Simulate the Markov chain.
    y_sim = simulate(grid, pmat, T, seed=2025)

    # Plot the simulated series.
    plt.figure(figsize=(10, 6))
    plt.plot(y_sim, marker='o', linestyle='-', color='b')
    plt.title("Simulated Markov Chain (50 Periods)")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.grid(True)
    plt.show()
