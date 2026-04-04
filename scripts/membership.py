import numpy as np
import matplotlib.pyplot as plt

def gaussian_mf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

# Features ke ranges (Actual values from your dataset)
# Onset_Duration (0 to 3500 approx)
x = np.linspace(0, 1, 100) # Normalized range

# Gaussian Parameters (Jo model ne seekhe honge)
mus = [0.2, 0.5, 0.8]   # Low, Medium, High
sigmas = [0.1, 0.1, 0.1]

plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red']
labels = ['Low', 'Medium', 'High']

for i in range(3):
    plt.plot(x, gaussian_mf(x, mus[i], sigmas[i]), color=colors[i], lw=3, label=labels[i])

plt.title('Initial Fuzzy Membership Functions for Clinical Inputs', fontsize=16, fontweight='bold')
plt.xlabel('Normalized Feature Value (0 to 1)', fontsize=12)
plt.ylabel('Membership Degree', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()