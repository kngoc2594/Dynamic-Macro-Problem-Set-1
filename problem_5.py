"""Problem 5

d) Numerically solve and simulate your model in (c). Plot your simulated data and compare them with
the actual data. You may try diferent combinations of parameters to make the simulated data look
like the actual data.
"""

import numpy as np
import matplotlib.pyplot as plt
# Stimualated data:
# Define parameters
alpha = 0.3          # Capital share in output
delta = 0.05         # Depreciation rate of capital
beta = 0.96          # Discount factor
theta = 2            # Relative risk aversion
gamma = 0.1          # Efficiency of human capital accumulation
rho = 0.98           # Persistence parameter for TFP
sigma = 0.02         # Standard deviation of shock
g_A = 0.02           # Growth rate of total factor productivity

# Initial conditions
K_0 = 10             # Initial physical capital
H_0 = 5              # Initial human capital
A_0 = 1              # Initial total factor productivity

# Time horizon
T = 100              # Number of periods to simulate
random_shocks = np.random.normal(0, sigma, T)  # Stochastic shocks

# Arrays to store results
K = np.zeros(T)
H = np.zeros(T)
A = np.zeros(T)
Y = np.zeros(T)
C = np.zeros(T)
I_K = np.zeros(T)
I_H = np.zeros(T)

# Initial values
K[0] = K_0
H[0] = H_0
A[0] = A_0

# Simulation loop
for t in range(1, T):
    # Total factor productivity shock
    A[t] = rho * A[t-1] + sigma * random_shocks[t-1]  # Stochastic process for A

    # Output production function
    Y[t-1] = A[t-1] * K[t-1]**alpha * (H[t-1])**(1-alpha)

    # Optimal investment in capital and human capital based on steady-state growth assumptions
    g_K = 0.02  # Assumed growth rate for capital (steady state)
    g_H = 0.02  # Assumed growth rate for human capital (steady state)

    # Optimal investment in capital (simple growth model assumption)
    I_K[t-1] = (g_K + delta) * K[t-1]

    # Optimal investment in human capital (simple growth model assumption)
    I_H[t-1] = g_H * H[t-1]

    # Capital and human capital accumulation
    K[t] = (1 - delta) * K[t-1] + I_K[t-1]
    H[t] = H[t-1] + I_H[t-1]

    # Consumption is the remainder after investment
    C[t-1] = Y[t-1] - I_K[t-1] - I_H[t-1]

# Plot the results
plt.figure(figsize=(12, 8))

# Plot for Capital (K), Human Capital (H), and Output (Y)
plt.subplot(2, 1, 2)
plt.plot(range(T), H, label='Human Capital (H)', color='green')
plt.xlabel('Time Period')
plt.ylabel('Human Capital (H)')
plt.title('Stimulated Human Capital Accumulation Over Time')
plt.grid(True)

plt.tight_layout()
plt.show()


#Actual data:
H_actual = np.array([29.919, 120.889, 297.1])

# Time periods for plotting
time_periods = np.array([0, 20, 40])  #20-year intervals for the 3 data points

# Plot the actual data
plt.figure(figsize=(12, 8))

# Plot Human Capital (Actual Data)
plt.subplot(2, 1, 1)
plt.plot(time_periods, H_actual, label='Actual Human Capital', color='green')
plt.xlabel('Time Period (years)')
plt.ylabel('Human Capital')
plt.title('Actual Human Capital Over Time')
plt.legend()

plt.tight_layout()
plt.show()

"""e) Conduct a counterfactual exercise that simulates growth policy that targets any mechanism described in (b). Make sure to describe the policy and compare the baseline results with the policy experiment.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Parameters for the model
alpha = 0.35         # Capital share in output
delta = 0.05         # Depreciation rate of capital
beta = 0.98          # Discount factor
gamma = 0.15         # Efficiency of human capital accumulation
rho = 0.98           # Persistence parameter for TFP
sigma = 0.05         # Standard deviation of shock
eta_K = 0.1         # Technology shock effect on physical capital
eta_H = 0.1         # Technology shock effect on human capital

K_0 = 50             # Initial physical capital
H_0 = 40             # Initial human capital
A_0 = 1.1            # Initial total factor productivity

T = 100              # Number of periods to simulate
random_shocks = np.random.normal(0, sigma, T)  # Stochastic shocks

# Arrays to store results
K = np.zeros(T)
H = np.zeros(T)
A_techshock = np.zeros(T)
Y_techshock = np.zeros(T)
C_techshock = np.zeros(T)
I_K_techshock = np.zeros(T)
I_H_techshock = np.zeros(T)

# Initial values
K[0] = K_0
H[0] = H_0
A_techshock[0] = A_0


# Simulation loop
for t in range(1, T):

    A_techshock[t] = rho * A_techshock[t-1] + sigma * random_shocks[t-1] + eta_K * K[t-1] + eta_H * H[t-1]
    Y_techshock[t-1] = A_techshock[t-1] * K[t-1]**alpha * H[t-1]**(1-alpha)  # Output with technology shock
    I_K_techshock[t-1] = (0.02 + delta) * K[t-1]
    I_H_techshock[t-1] = gamma * H[t-1]**0.5  # Nonlinear human capital investment
    K[t] = (1 - delta) * K[t-1] + I_K_techshock[t-1]
    H[t] = H[t-1] + I_H_techshock[t-1]
    C_techshock[t-1] = Y_techshock[t-1] - I_K_techshock[t-1] - I_H_techshock[t-1]

# Plot the results c
plt.figure(figsize=(12, 8))

# Plot for Human Capital (H) with technology shock
plt.subplot(2, 2, 1)
plt.plot(range(T), H, label='Human Capital (H) with Technology Shock', color='green')
plt.xlabel('Time Period')
plt.ylabel('Human Capital (H)')
plt.title('Human Capital Accumulation Over Time')
plt.legend()


# Plot for Capital (K) with technology shock
plt.subplot(2, 2, 2)
plt.plot(range(T), K, label='Capital (K) with Technology Shock', color='cyan')
plt.xlabel('Time Period')
plt.ylabel('Capital (K)')
plt.title('Capital Accumulation Over Time')
plt.legend()

plt.tight_layout()
plt.show()
