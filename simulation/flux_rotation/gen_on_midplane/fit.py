import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

deg = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
data = np.array([79111, 76867, 71534, 63827, 54111, 42693, 30096, 18904, 13673, 12081], dtype=float)

# Convert degree to radians
theta = np.radians(deg)

# Define the fitting function: A * cos^n + a
def cos_pow_plus_offset(theta, A, n, a):
    return A * np.cos(theta)**n + a

# Perform curve fit
popt, pcov = curve_fit(cos_pow_plus_offset, theta, data) #, bounds=([0, 0, -np.inf], [100, 100, np.inf]))

A_fit, n_fit, a_fit = popt

print(f"Fitted A = {A_fit:.3f}")
print(f"Fitted n = {n_fit:.3f}")
print(f"Fitted a = {a_fit:.3f}")

# Plot the results
theta_vals = np.linspace(0, np.pi/2, 500)
fit_vals = cos_pow_plus_offset(theta_vals, A_fit, n_fit, a_fit)

plt.scatter(deg, data, color='r', label='Data')
plt.plot(np.degrees(theta_vals), fit_vals, label=f'Fit: {A_fit:.3f} * cos^{n_fit:.3f}(theta) + {a_fit:.3f}')
plt.xlabel('Angle (deg)')
plt.ylabel('Data')
plt.legend()
plt.show()
