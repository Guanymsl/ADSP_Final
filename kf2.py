import numpy as np
import matplotlib.pyplot as plt

dt = 1.0
num_steps = 50

true_acceleration = np.array([0.1, -0.05]) # [ax, ay]
x_true_2d = np.zeros((4, num_steps))
x_true_2d[:, 0] = [0.0, 1.0, 0.0, 1.5]

Q_2d = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
measurement_noise_std = 1.0
R_2d = measurement_noise_std**2 * np.eye(2)

# Measurement matrix: we only measure position [x, y]
H_2d = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0]])

# State transition matrix
F_2d = np.array([[1, dt, 0,  0],
                 [0,  1, 0,  0],
                 [0,  0, 1, dt],
                 [0,  0, 0,  1]])

# Control input model
G_2d = np.array([[0.5*dt**2, 0],
                 [dt,        0],
                 [0, 0.5*dt**2],
                 [0,        dt]])

measurements_2d = np.zeros((2, num_steps))
for k in range(1, num_steps):
    x_true_2d[:, k] = F_2d.dot(x_true_2d[:, k-1]) + G_2d.dot(true_acceleration)
    x_true_2d[:, k] += np.random.multivariate_normal(np.zeros(4), Q_2d)
    measurements_2d[:, k] = H_2d.dot(x_true_2d[:, k]) + np.random.randn(2) * measurement_noise_std

# Kalman filter initialization
x_hat_2d = np.zeros((4, num_steps))
P_2d = np.eye(4)
x_hat_2d[:, 0] = [0, 0, 0, 0]

cov_trace = []

for k in range(1, num_steps):
    # Prediction step
    x_pred = F_2d.dot(x_hat_2d[:, k-1]) + G_2d.dot(true_acceleration)
    P_pred = F_2d.dot(P_2d).dot(F_2d.T) + Q_2d

    # Update step
    z = measurements_2d[:, k]
    y = z - H_2d.dot(x_pred)
    S = H_2d.dot(P_pred).dot(H_2d.T) + R_2d
    K = P_pred.dot(H_2d.T).dot(np.linalg.inv(S))
    x_hat_2d[:, k] = x_pred + K.dot(y)
    P_2d = (np.eye(4) - K.dot(H_2d)).dot(P_pred)

    # Store trace of covariance matrix
    cov_trace.append(np.trace(P_2d))

plt.figure()
plt.plot(x_true_2d[0], x_true_2d[2], label='True Position')
plt.plot(measurements_2d[0], measurements_2d[1], '.', label='Measurements')
plt.plot(x_hat_2d[0], x_hat_2d[2], label='KF Estimate')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Newtonian Motion with Kalman Filter')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.figure()
plt.plot(range(1, num_steps), cov_trace)
plt.xlabel('Time Step')
plt.ylabel('Trace of Covariance Matrix')
plt.title('Uncertainty Over Time (Trace of Covariance)')
plt.grid(True)
plt.show()
