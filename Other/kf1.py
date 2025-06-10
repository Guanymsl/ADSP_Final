import numpy as np
import matplotlib.pyplot as plt

dt = 1.0
num_steps = 50

true_acceleration = 0.1
x_true = np.zeros((2, num_steps)) # [position, velocity]
x_true[:, 0] = [0.0, 1.0]

process_noise_cov = np.array([[1e-4, 0], [0, 1e-4]])
measurement_noise_std = 1.0

measurements = np.zeros(num_steps)
for k in range(1, num_steps):
    # True motion update: x_k = F * x_{k-1} + G * a
    F = np.array([[1, dt], [0, 1]])
    G = np.array([0.5 * dt**2, dt])
    x_true[:, k] = F.dot(x_true[:, k-1]) + G * true_acceleration
    # Add process noise
    x_true[:, k] += np.random.multivariate_normal([0,0], process_noise_cov)
    # Measurement: position only
    measurements[k] = x_true[0, k] + np.random.randn() * measurement_noise_std

# Kalman filter initialization
x_hat = np.zeros((2, num_steps))
P = np.eye(2)
Q = process_noise_cov
R = measurement_noise_std**2
H = np.array([[1, 0]])

x_hat[:, 0] = [0.0, 0.0]  # initial estimate

# Kalman filter loop
for k in range(1, num_steps):
    # Prediction
    x_pred = F.dot(x_hat[:, k-1]) + G * true_acceleration
    P_pred = F.dot(P).dot(F.T) + Q

    # Measurement update
    z = measurements[k]
    y = z - H.dot(x_pred)
    S = H.dot(P_pred).dot(H.T) + R
    K = P_pred.dot(H.T) / S
    x_hat[:, k] = x_pred + K.flatten() * y
    P = (np.eye(2) - K.dot(H)).dot(P_pred)

plt.figure()
plt.plot(x_true[0], label='True Position')
plt.plot(measurements, label='Measurements')
plt.plot(x_hat[0], label='KF Estimate')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('1D Newtonian Motion with Kalman Filter')
plt.legend()
plt.grid(True)