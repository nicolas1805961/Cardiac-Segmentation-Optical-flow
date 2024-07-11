import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

# Generate input values
x = np.linspace(-4, 3, 400)

# Compute the activation values
y_relu = relu(x)
y_gelu = gelu(x)
y_leaky_relu = leaky_relu(x)

# Plotting the activation functions
plt.figure(figsize=(10, 6))

plt.plot(x, y_relu, label='ReLU', linewidth=2)
plt.plot(x, y_gelu, label='GELU', linewidth=2)
plt.plot(x, y_leaky_relu, label='Leaky ReLU', linewidth=2)

plt.title('Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.ylim([-1, 3])
plt.xlim([-4, 3])
plt.tight_layout()

# Show the plot
plt.savefig('activation_function.png', dpi=600) #dpi=600