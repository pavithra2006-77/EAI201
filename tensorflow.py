import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# User input for slope and intercept
m = float(input("Enter slope (m): "))
c = float(input("Enter intercept (c): "))

# Generate data points with noise
X = np.linspace(0, 20, 100)  # more points and wider range
Y = m * X + c + np.random.randn(*X.shape) * 1.5  # slightly higher noise

# Build a simple linear regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

# Compile the model with custom learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
              loss='mean_squared_error')

# Train the model
print("Training...")
model.fit(X, Y, epochs=80, verbose=2)  # verbose=2 shows some progress
print("Training done!")

# Extract learned slope and intercept
w, b = model.layers[0].get_weights()
learned_slope = w[0][0]
learned_intercept = b[0]
print(f"Learned line: y = {learned_slope:.2f}x + {learned_intercept:.2f}")

# Plot data and fitted line
plt.figure(figsize=(8,5))
plt.scatter(X, Y, color='blue', alpha=0.6, label="Noisy Data", marker='o')
predicted_Y = model.predict(X.reshape(-1, 1))
plt.plot(X, predicted_Y, color="red", linewidth=2, label="Fitted Line")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("TensorFlow Linear Regression (Modified)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
