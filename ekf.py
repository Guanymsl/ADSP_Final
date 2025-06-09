import numpy as np
import matplotlib.pyplot as plt

# Step size
dt = 0.1
N = 10000

# Declaration
x_ideal = np.zeros((3, N))
x_true  = np.zeros((3, N))
x_est = np.zeros((3, N))
P = np.zeros((3, 3, N))
u = np.zeros((2, N))
z = np.zeros(N)

# For covariance plotting
covariance_trace = np.zeros(N)

x_ideal[:, 0] = [0, 0, 0]
x_true[:, 0] = [0, 0, 0]
x_est[:, 0] = [0.1, 0.1, 0.1]

P[:, :, 0] = np.eye(3) * 0.1
z[0] = 0.1

for k in range(N):
    u[:, k] = [0.1, np.deg2rad(0.2)]

sigma_v = 0.001
sigma_omega = np.deg2rad(0.01)
Q = np.diag([sigma_v**2, sigma_v**2, sigma_omega**2])

sigma_r = 1
R = np.array([[sigma_r**2]])

for k in range(1, N):
    v = u[0, k-1]
    omega = u[1, k-1]

    # Ideal trajectory
    theta_ideal_prev = x_ideal[2, k-1]
    vx_ideal = v * np.cos(theta_ideal_prev)
    vy_ideal = v * np.sin(theta_ideal_prev)
    x_ideal[0, k] = x_ideal[0, k-1] + vx_ideal * dt
    x_ideal[1, k] = x_ideal[1, k-1] + vy_ideal * dt
    x_ideal[2, k] = (x_ideal[2, k-1] + omega * dt + np.pi) % (2 * np.pi) - np.pi

    # True trajectory with noise
    theta_true_prev  = x_true[2, k-1]
    vx = v * np.cos(theta_true_prev)
    vy = v * np.sin(theta_true_prev)
    x_true[0, k] = x_true[0, k-1] + vx * dt + np.random.normal(0, sigma_v)
    x_true[1, k] = x_true[1, k-1] + vy * dt + np.random.normal(0, sigma_v)
    x_true[2, k] = (x_true[2, k-1] + omega * dt + np.random.normal(0, sigma_omega) + np.pi) % (2 * np.pi) - np.pi

    # Range measurement
    z[k] = np.sqrt(x_true[0, k]**2 + x_true[1, k]**2) + np.random.normal(0, sigma_r)

    # EKF prediction
    x_prev = x_est[:, k-1].copy()
    P_prev = P[:, :, k-1].copy()
    theta_est_prev = x_prev[2]

    x_pred = np.array([
        x_prev[0] + v * np.cos(theta_est_prev) * dt,
        x_prev[1] + v * np.sin(theta_est_prev) * dt,
        x_prev[2] + omega * dt
    ])

    A = np.eye(3)
    A[0, 2] = -v * np.sin(theta_est_prev) * dt
    A[1, 2] =  v * np.cos(theta_est_prev) * dt

    P_pred = A @ P_prev @ A.T + Q

    # EKF update
    hx = np.sqrt(x_pred[0]**2 + x_pred[1]**2)
    H = np.array([[x_pred[0]/hx, x_pred[1]/hx, 0.0]])
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    zk = z[k]
    x_upd = x_pred + (K.flatten() * (zk - hx))
    x_upd[2] = (x_upd[2] + np.pi) % (2 * np.pi) - np.pi

    P_upd = (np.eye(3) - K @ H) @ P_pred

    x_est[:, k] = x_upd
    P[:, :, k] = P_upd

    # Compute covariance trace (x and y only)
    covariance_trace[k] = P_upd[0, 0] + P_upd[1, 1]

# Plot trajectories
plt.figure(figsize=(6, 6))
plt.plot(x_ideal[0, :], x_ideal[1, :], 'g-', linewidth=2, label='Ideal (noiseless)')
plt.plot(x_true[0, :],  x_true[1, :],  'b-', linewidth=1, label='Noisy Truth')
plt.plot(x_est[0, :],   x_est[1, :],   'r--', linewidth=2, label='EKF Estimate')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trajectories: Ideal vs. Noisy Truth vs. EKF')
plt.legend(loc='best')
plt.grid(True)
plt.axis('equal')

# Plot covariance trace
time = np.arange(N) * dt
plt.figure(figsize=(6, 4))
plt.plot(time, covariance_trace, linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Trace of Covariance (m²)')
plt.title('Trace of Position Covariance vs. Time')
plt.grid(True)

plt.show()
